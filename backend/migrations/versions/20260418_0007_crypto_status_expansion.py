"""expand provenance verification statuses"""

from alembic import op


revision = "20260418_0007"
down_revision = "20260321_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE provenanceverificationstatus ADD VALUE IF NOT EXISTS 'partial_provenance'")
    op.execute("ALTER TYPE provenanceverificationstatus ADD VALUE IF NOT EXISTS 'no_provenance'")
    op.execute("ALTER TYPE provenanceverificationstatus ADD VALUE IF NOT EXISTS 'issuer_unknown'")


def downgrade() -> None:
    # PostgreSQL enum value removal is intentionally omitted in downgrade for safety.
    pass
