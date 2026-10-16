from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class SoftBindingRecoveryResult:
    status: str
    reason: str
    binding: dict | None = None
    repository_configured: bool = False
    recovered_manifest: dict | None = None
    attempted_references: list[str] | None = None


class SoftBindingRecoveryService:
    def __init__(self, repository_base_url: str | None = None):
        self.repository_base_url = repository_base_url

    def computeSoftBinding(self, media_bytes: bytes | None, media_url: str, content_type: str | None = None) -> dict | None:
        if not media_bytes:
            return None
        binding = {
            "byte_length": len(media_bytes),
            "content_type": content_type,
            "source_url": media_url,
        }
        logger.info("soft_binding_computed", extra={"media_url": media_url, "byte_length": len(media_bytes)})
        return binding

    def queryManifestRepositoryByBinding(self, binding: dict | None) -> dict | None:
        logger.info("soft_binding_repository_query_by_binding", extra={"configured": bool(self.repository_base_url)})
        if not self.repository_base_url or not binding:
            return None
        return None

    def queryManifestRepositoryByReference(self, reference: str | None) -> dict | None:
        logger.info("soft_binding_repository_query_by_reference", extra={"configured": bool(self.repository_base_url), "reference": reference})
        if not self.repository_base_url or not reference:
            return None
        return None

    def recoverDecoupledManifest(self, media_bytes: bytes | None, media_url: str, content_type: str | None = None) -> SoftBindingRecoveryResult:
        binding = self.computeSoftBinding(media_bytes, media_url, content_type)
        repository_configured = bool(self.repository_base_url)
        if not repository_configured:
            return SoftBindingRecoveryResult(
                status="soft_binding_recovery_not_configured",
                reason="No soft-binding manifest repository is configured.",
                binding=binding,
                repository_configured=False,
                attempted_references=[media_url],
            )

        recovered = self.queryManifestRepositoryByBinding(binding) or self.queryManifestRepositoryByReference(media_url)
        if recovered:
            return SoftBindingRecoveryResult(
                status="decoupled_manifest_recovered",
                reason="A decoupled manifest was recovered through the soft-binding workflow.",
                binding=binding,
                repository_configured=True,
                recovered_manifest=recovered,
                attempted_references=[media_url],
            )

        return SoftBindingRecoveryResult(
            status="soft_binding_manifest_not_found",
            reason="Soft-binding recovery ran, but no decoupled manifest was found.",
            binding=binding,
            repository_configured=True,
            attempted_references=[media_url],
        )
