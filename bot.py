import asyncio
import logging
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hotd-bot")


class HotDBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        # We do not require message content for slash commands and components
        super().__init__(command_prefix=commands.when_mentioned_or("!"), intents=intents)

    async def setup_hook(self) -> None:
        await self.load_extension("cogs.hotd_rp")
        try:
            await self.tree.sync()
            logger.info("App commands synced globally")
        except Exception:
            logger.exception("Failed to sync app commands")


async def main() -> None:
    load_dotenv()
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set. Put it in your environment or a .env file.")

    bot = HotDBot()

    @bot.event
    async def on_ready():
        logger.info("Logged in as %s (%s)", bot.user, bot.user and bot.user.id)

    await bot.start(token)


if __name__ == "__main__":
    try:
        import uvloop  # type: ignore

        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())

