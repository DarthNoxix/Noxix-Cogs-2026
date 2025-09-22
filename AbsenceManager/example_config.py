# Example configuration for AbsenceManager cog
# This file shows how to customize the cog's behavior

# Default configuration values
DEFAULT_CONFIG = {
    "enabled": False,
    "channel_id": None,
    "message_id": None,
    "authorized_roles": [],
    "embed_title": "📋 Staff Absences",
    "embed_color": 0x3498db,  # Blue color
    "show_avatar": True,
    "auto_remove_expired": True,
    "expired_grace_period": 24,  # hours
}

# Example custom configurations for different server types

# Gaming Server Configuration
GAMING_CONFIG = {
    "embed_title": "🎮 Staff Availability",
    "embed_color": 0x9b59b6,  # Purple
    "show_avatar": True,
    "auto_remove_expired": True,
    "expired_grace_period": 12,  # Shorter grace period for gaming
}

# Business/Professional Server Configuration
BUSINESS_CONFIG = {
    "embed_title": "📅 Team Availability",
    "embed_color": 0x2ecc71,  # Green
    "show_avatar": False,  # More professional without avatars
    "auto_remove_expired": True,
    "expired_grace_period": 48,  # Longer grace period for business
}

# Community Server Configuration
COMMUNITY_CONFIG = {
    "embed_title": "👥 Staff Status",
    "embed_color": 0xe67e22,  # Orange
    "show_avatar": True,
    "auto_remove_expired": False,  # Keep expired for reference
    "expired_grace_period": 72,  # Very long grace period
}

# Example usage in a custom cog or setup script:
"""
# To apply a custom configuration:
await bot.get_cog("AbsenceManager").config.guild(guild).embed_title.set("🎮 Staff Availability")
await bot.get_cog("AbsenceManager").config.guild(guild).embed_color.set(0x9b59b6)
await bot.get_cog("AbsenceManager").config.guild(guild).show_avatar.set(True)
"""
