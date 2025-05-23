"""create generated_models table

Revision ID: 9b867bb3f035
Revises: 5a271b80e680
Create Date: 2025-05-06 19:12:59.747268

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9b867bb3f035'
down_revision: Union[str, None] = '5a271b80e680'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('generated_models',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('default', sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('generated_models')
    # ### end Alembic commands ###
