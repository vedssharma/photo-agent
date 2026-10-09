"""Content Credentials (C2PA) for exports that used generative edits.

When any generative edit is visible in an export, the file is signed with a C2PA manifest
saying the photo was edited with generative AI (IPTC digital source type
`compositeWithTrainedAlgorithmicMedia`), listing each generative edit and the model that
made it. Viewers such as https://contentcredentials.org/verify show it.

The signing certificate is made locally on first use (`<data_dir>/c2pa/`): a small
certificate authority and a signing certificate under it. Verifiers will read the
manifest but report the signer as unknown, since that authority is not on any trust list;
a product would sign with a certificate from a C2PA-trusted issuer instead.
"""

from __future__ import annotations

import datetime
import io
import json
import threading
from pathlib import Path
from typing import Any

import c2pa
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from photo_agent.layers import EditState
from photo_agent.operations import GenerativeBase

APP = "photo-agent"
VERSION = "0.1.0"
GENERATED = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
CAPTURED = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
MIME = {"jpeg": "image/jpeg", "png": "image/png"}

_lock = threading.Lock()


def visible_generative_ops(state: EditState) -> list[GenerativeBase]:
    """Generative operations that show in a render of this state."""
    shown: list[GenerativeBase] = [op for op in state.framing if isinstance(op, GenerativeBase)]
    for layer in state.layers:
        if layer.visible and layer.opacity > 0:
            shown += [op for op in layer.operations if isinstance(op, GenerativeBase)]
    return shown


def manifest(state: EditState, title: str) -> dict[str, Any]:
    edits = visible_generative_ops(state)
    actions: list[dict[str, Any]] = [{"action": "c2pa.created", "digitalSourceType": CAPTURED}]
    for op in edits:
        actions.append(
            {
                "action": "c2pa.edited",
                "digitalSourceType": GENERATED,
                "softwareAgent": {"name": APP, "version": VERSION},
                "description": op.summary(),
                "parameters": {"model": op.model or "unknown"},
            }
        )
    return {
        "claim_generator_info": [{"name": APP, "version": VERSION}],
        "title": title,
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
    }


def sign(data: bytes, fmt: str, state: EditState, title: str, data_dir: Path) -> bytes:
    """`data` (an encoded JPEG or PNG) with Content Credentials for its generative edits."""
    builder = c2pa.Builder(json.dumps(manifest(state, title)))
    out = io.BytesIO()
    with _lock:
        builder.sign(_signer(data_dir), MIME[fmt], io.BytesIO(data), out)
    return out.getvalue()


def _signer(data_dir: Path) -> c2pa.Signer:
    folder = data_dir / "c2pa"
    chain, key = folder / "chain.pem", folder / "signing-key.pem"
    if not (chain.exists() and key.exists()):
        folder.mkdir(parents=True, exist_ok=True)
        cert_pem, key_pem = _make_certificates()
        key.write_bytes(key_pem)
        key.chmod(0o600)
        chain.write_bytes(cert_pem)
    info = c2pa.C2paSignerInfo(
        alg=b"es256", sign_cert=chain.read_bytes(), private_key=key.read_bytes(), ta_url=None
    )
    return c2pa.Signer.from_info(info)


def _name(common: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, f"{APP} (local)"),
        ]
    )


def _usage(sign: bool) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=sign,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=not sign,
        crl_sign=not sign,
        encipher_only=False,
        decipher_only=False,
    )


def _make_certificates() -> tuple[bytes, bytes]:
    """A local CA and a signing certificate under it, as C2PA requires (an end-entity
    certificate with an email-protection key purpose, not self-signed)."""
    now = datetime.datetime.now(datetime.UTC)
    start, end = now - datetime.timedelta(days=1), now + datetime.timedelta(days=3650)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (
        x509.CertificateBuilder()
        .subject_name(_name(f"{APP} local CA"))
        .issuer_name(_name(f"{APP} local CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_usage(sign=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name(APP))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_usage(sign=True), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False)
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    chain = leaf.public_bytes(pem) + ca.public_bytes(pem)
    private = key.private_bytes(
        pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    return chain, private
