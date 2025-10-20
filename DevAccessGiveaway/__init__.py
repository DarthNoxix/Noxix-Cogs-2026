from redbot.core.utils import get_end_user_data_statement
from redbot.core.bot import Red

from .devaccessgiveaway import DevAccessGiveaway

__red_end_user_data_statement__ = get_end_user_data_statement(file=__file__)


async def setup(bot: Red) -> None:
    await bot.add_cog(DevAccessGiveaway(bot))
