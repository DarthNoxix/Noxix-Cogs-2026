from .absencemanager import AbsenceManager

__red_end_user_data_statement__ = (
    "This cog stores absence records and configuration data for your server. "
    "Absence data includes user IDs, absence reasons, and return dates. "
    "This data is stored locally and can be deleted by server administrators."
)


def setup(bot):
    bot.add_cog(AbsenceManager(bot))
