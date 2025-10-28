## House of the Dragon RP Discord Bot

Bring Westeros to your server with a rich, button-and-modal powered RP hub.

### Features

- Persistent hub panel with:
  - Create Character modal
  - House selector (Targaryen, Velaryon, Hightower, Strong, etc.)
  - Start Scene modal (creates thread, public/private)
  - Request Duel modal (creates duel thread)
  - Claim Daily coins and Leaderboard
  - Lore link button
- Slash commands under `/hotd`:
  - `/hotd post_panel` — Post the RP hub panel in a channel
  - `/hotd config scene_private:true|false` — Default privacy for new scenes
  - `/hotd roll 2d6+3` — Simple dice roller
  - `/hotd character [@user]` — Show a character card
  - `/hotd daily`, `/hotd balance` — Coin utilities
- SQLite persistence. Safe to restart; buttons are persistent.

### Setup

1. Requirements

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure your bot token

   Create a `.env` file:

   ```bash
   echo "DISCORD_TOKEN=your_token_here" > .env
   ```

3. Run the bot

   ```bash
   python bot.py
   ```

4. Post the hub panel

   In your server, run `/hotd post_panel` in the channel where you want the hub. Use `/hotd config` to make scenes default to private threads if desired.

### Notes

- The bot stores data in `data/hotd.db`. Change location by setting `HOTD_DATA_DIR`.
- Make sure your bot has the permissions to create threads and use application commands.

Just useful cogs for my bot for ADOD and more
