from .websitesubs import WebsiteSubs

__red_end_user_data_statement__ = (
    "This cog stores user subscription data including subscription dates, tiers, and expiration information."
)

async def setup(bot):
    cog = WebsiteSubs(bot)
    await bot.add_cog(cog)
