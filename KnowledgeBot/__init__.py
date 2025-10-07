from .knowledgebot import KnowledgeBot

__red_end_user_data_statement__ = "This cog stores user feedback and categorization data for support ticket routing."


async def setup(bot):
    await bot.add_cog(KnowledgeBot(bot))
