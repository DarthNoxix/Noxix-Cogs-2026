import discord
from discord.ui import View, Button, Select
from typing import List, Dict, Any
from redbot.core import commands


class MemoryViewer(View):
    """View for browsing and managing memories"""
    
    def __init__(self, memories: Dict[str, Any], timeout: int = 60):
        super().__init__(timeout=timeout)
        self.memories = memories
        self.current_page = 0
        self.items_per_page = 10
        
        # Create buttons with proper callbacks
        self.prev_button = Button(label="◀️", style=discord.ButtonStyle.secondary)
        self.next_button = Button(label="▶️", style=discord.ButtonStyle.secondary)
        self.delete_button = Button(label="🗑️", style=discord.ButtonStyle.danger)
        
        # Set up callbacks
        self.prev_button.callback = self.previous_page
        self.next_button.callback = self.next_page
        self.delete_button.callback = self.delete_memory
        
        # Add buttons to view
        self.add_item(self.prev_button)
        self.add_item(self.next_button)
        self.add_item(self.delete_button)
        
    async def get_embed(self) -> discord.Embed:
        """Get the current page embed"""
        embed = discord.Embed(
            title="OpenWebUI Memories",
            color=discord.Color.blue()
        )
        
        if not self.memories:
            embed.description = "No memories found."
            return embed
        
        # Get current page items
        memory_items = list(self.memories.items())
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_items = memory_items[start_idx:end_idx]
        
        for name, data in page_items:
            text = data.get("text", "No text")[:100]
            embed.add_field(
                name=name,
                value=f"{text}..." if len(text) == 100 else text,
                inline=False
            )
        
        total_pages = (len(memory_items) + self.items_per_page - 1) // self.items_per_page
        embed.set_footer(text=f"Page {self.current_page + 1}/{total_pages}")
        
        return embed
    
    async def previous_page(self, interaction: discord.Interaction):
        """Handle previous page button click"""
        if self.current_page > 0:
            self.current_page -= 1
            embed = await self.get_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()
    
    async def next_page(self, interaction: discord.Interaction):
        """Handle next page button click"""
        total_pages = (len(self.memories) + self.items_per_page - 1) // self.items_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            embed = await self.get_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()
    
    async def delete_memory(self, interaction: discord.Interaction):
        """Handle delete memory button click"""
        # This would need to be implemented with a modal or select menu
        await interaction.response.send_message("Delete functionality would be implemented here.", ephemeral=True)


class ModelSelector(Select):
    """Select menu for choosing models"""
    
    def __init__(self, models: List[str], current_model: str = None):
        options = []
        for model in models[:25]:  # Discord limit
            options.append(discord.SelectOption(
                label=model,
                value=model,
                default=model == current_model
            ))
        super().__init__(placeholder="Select a model...", options=options)
    
    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Selected model: {self.values[0]}", ephemeral=True)


class EmbeddingModelSelector(Select):
    """Select menu for choosing embedding models"""
    
    def __init__(self, models: List[str], current_model: str = None):
        options = []
        for model in models[:25]:  # Discord limit
            options.append(discord.SelectOption(
                label=model,
                value=model,
                default=model == current_model
            ))
        super().__init__(placeholder="Select an embedding model...", options=options)
    
    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Selected embedding model: {self.values[0]}", ephemeral=True)


class SettingsView(View):
    """View for managing assistant settings"""
    
    def __init__(self, conf: Any, timeout: int = 60):
        super().__init__(timeout=timeout)
        self.conf = conf
        
        # Create refresh button with proper callback
        self.refresh_button = Button(label="🔄 Refresh", style=discord.ButtonStyle.primary)
        self.refresh_button.callback = self.refresh
        
        # Add button to view
        self.add_item(self.refresh_button)
    
    async def get_embed(self) -> discord.Embed:
        """Get the settings embed"""
        embed = discord.Embed(
            title="OpenWebUI Assistant Settings",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="Model Settings",
            value=f"**Chat Model:** {self.conf.model}\n"
                  f"**Embedding Model:** {self.conf.embed_model}\n"
                  f"**Temperature:** {self.conf.temperature}\n"
                  f"**Max Tokens:** {self.conf.max_tokens}",
            inline=True
        )
        
        embed.add_field(
            name="Auto-Response Settings",
            value=f"**Enabled:** {self.conf.enabled}\n"
                  f"**Channel ID:** {self.conf.channel_id or 'None'}\n"
                  f"**Min Length:** {self.conf.min_length}\n"
                  f"**Question Only:** {self.conf.endswith_questionmark}\n"
                  f"**Mention Only:** {self.conf.mention}",
            inline=True
        )
        
        embed.add_field(
            name="Memory Settings",
            value=f"**Memories Count:** {len(self.conf.embeddings)}\n"
                  f"**Top N:** {self.conf.top_n}\n"
                  f"**Min Relatedness:** {self.conf.min_relatedness}",
            inline=True
        )
        
        return embed
    
    async def refresh(self, interaction: discord.Interaction):
        """Handle refresh button click"""
        embed = await self.get_embed()
        await interaction.response.edit_message(embed=embed, view=self)
