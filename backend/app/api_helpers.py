from app.schemas import AuditEntryResponse, HistoryItemResponse, HistoryResponse, VerificationResultResponse


def job_to_response(job) -> VerificationResultResponse:
    return VerificationResultResponse(
        job_id=job.id,
        input_type=job.input_type.value,
        status=job.status,
        headline_message=job.headline_message,
        short_reason=job.short_reason,
        content_preview=job.content_preview or {},
        proof_summary=job.proof_summary or {},
        analysis_report=job.analysis_report or {},
        layer_results=job.layer_results or {},
        audit_trail=[
            AuditEntryResponse(
                event_type=entry.event_type,
                event_payload=entry.event_payload,
                previous_hash=entry.previous_hash,
                entry_hash=entry.entry_hash,
                created_at=entry.created_at,
            )
            for entry in sorted(job.audit_entries, key=lambda item: item.created_at)
        ],
        created_at=job.created_at,
    )


def history_to_response(jobs) -> HistoryResponse:
    return HistoryResponse(
        items=[
            HistoryItemResponse(
                job_id=job.id,
                status=job.status,
                input_type=job.input_type.value,
                headline_message=job.headline_message,
                created_at=job.created_at,
            )
            for job in jobs
        ]
    )
