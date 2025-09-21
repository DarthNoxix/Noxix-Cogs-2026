from .botbackup import BotBackup

__red_end_user_data_statement__ = "This cog stores backup files of bot configurations. These backups may contain server settings and user data from other cogs."


def setup(bot):
    bot.add_cog(BotBackup(bot))
