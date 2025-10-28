import json
import os
import threading
from typing import Any, Dict, Optional


class JSONStorage:
    """Simple thread-safe JSON storage for profiles and state.

    Files:
      - profiles: maps user_id (str) -> profile dict
      - state: holds per-guild panel message mapping
        {
          "panels": {
            "<guild_id>": {"channel_id": int, "message_id": int}
          }
        }
    """

    def __init__(self, data_directory: str = "/workspace/data") -> None:
        # Allow override via HOTD_DATA_DIR; fallback to provided default
        env_dir = os.environ.get("HOTD_DATA_DIR")
        self.data_directory = env_dir or data_directory
        os.makedirs(self.data_directory, exist_ok=True)

        self.profiles_path = os.path.join(self.data_directory, "hotd_profiles.json")
        self.state_path = os.path.join(self.data_directory, "hotd_state.json")

        self._lock = threading.RLock()

        self._ensure_file(self.profiles_path, default={})
        self._ensure_file(self.state_path, default={"panels": {}})

    # ---------- file helpers ----------
    def _ensure_file(self, path: str, default: Any) -> None:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2, ensure_ascii=False)

    def _read_json(self, path: str) -> Any:
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _write_json(self, path: str, data: Any) -> None:
        with self._lock:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)

    # ---------- profiles API ----------
    def get_profiles(self) -> Dict[str, Dict[str, Any]]:
        data: Dict[str, Dict[str, Any]] = self._read_json(self.profiles_path)
        return data

    def get_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        user_key = str(user_id)
        data = self.get_profiles()
        return data.get(user_key)

    def upsert_profile(self, user_id: int, profile: Dict[str, Any]) -> Dict[str, Any]:
        user_key = str(user_id)
        data = self.get_profiles()
        data[user_key] = profile
        self._write_json(self.profiles_path, data)
        return profile

    def delete_profile(self, user_id: int) -> None:
        user_key = str(user_id)
        data = self.get_profiles()
        if user_key in data:
            del data[user_key]
            self._write_json(self.profiles_path, data)

    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        return self.get_profiles()

    # ---------- state API ----------
    def get_state(self) -> Dict[str, Any]:
        return self._read_json(self.state_path)

    def get_panel_state(self, guild_id: int) -> Optional[Dict[str, int]]:
        guild_key = str(guild_id)
        state = self.get_state()
        panels: Dict[str, Dict[str, int]] = state.get("panels", {})
        return panels.get(guild_key)

    def set_panel_state(self, guild_id: int, channel_id: int, message_id: int) -> Dict[str, Any]:
        guild_key = str(guild_id)
        state = self.get_state()
        panels: Dict[str, Dict[str, int]] = state.get("panels", {})
        panels[guild_key] = {"channel_id": int(channel_id), "message_id": int(message_id)}
        state["panels"] = panels
        self._write_json(self.state_path, state)
        return panels[guild_key]

