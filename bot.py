import asyncio
import logging
import os
from typing import Optional

import discord
from discord.ext import commands


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hotd-bot")


def get_token() -> Optional[str]:
    token = os.getenv("DISCORD_TOKEN") or os.getenv("BOT_TOKEN")
    return token


intents = discord.Intents.default()
intents.guilds = True
intents.members = True
intents.message_content = False


class HotDBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(
            command_prefix=commands.when_mentioned_or("!"),
            intents=intents,
            help_command=None,
        )

    async def setup_hook(self) -> None:
        await self.load_extension("cogs.hotd_rp")
        try:
            synced = await self.tree.sync()
            logger.info("Synced %d application commands", len(synced))
        except Exception as exc:
            logger.exception("Failed to sync commands: %s", exc)


async def main() -> None:
    token = get_token()
    if not token:
        raise RuntimeError("Set DISCORD_TOKEN in environment.")
    bot = HotDBot()
    async with bot:
        await bot.start(token)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

