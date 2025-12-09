from copy import deepcopy
from pathlib import Path
from typing import Literal
import aiofiles
import ujson

work_dir = Path(__file__).parent / "data" / "state_diff"
work_dir.mkdir(parents=True, exist_ok=True)

type JsonSerializable = str | int | float | bool | None | list[JsonSerializable] | dict[str, JsonSerializable]


class StateDiff:
    
    def __init__(self, app_id: str, user_id: str, maximum_entries: int=500) -> None:
        self.app_id: str = app_id
        self.user_id: str = user_id
        self.maximum_entries: int = maximum_entries
        self.state_file: Path = work_dir / f"{app_id}-{user_id}.json"
        self.state: list[JsonSerializable] = []
        if self.state_file.exists():
            self.state = ujson.loads(self.state_file.read_text())
        
    async def write(self, state: list[JsonSerializable]) -> None:
        """Write a state to the file."""
        async with aiofiles.open(self.state_file, "w") as f:
            _ = await f.write(ujson.dumps(state))
        
    async def diff(self, new_state: list[JsonSerializable], mode: Literal["diff", "incremental"]) -> list[JsonSerializable]:
        """Compute the diff between the current state and the new state.
        
        mode 'diff': return all changed items (new or removed), and replace the old state.
        mode 'incremental': return all new items not in previous state, and append new items to state.
        If no previous state exists, treat all new entries as new.
        
        An empty list is returned if no new state is found.
        """
        if not self.state:  # No previous state, treat all entries as new
            await self.write(new_state)
            self.state = new_state
            return new_state
        
        # Previous state exists, compute diff
        returned_data: list[JsonSerializable] = []
        
        old_state = deepcopy(self.state)
        # Reverse traversal and pop
        for item_idx in range(len(new_state) - 1, -1, -1):
            item = new_state[item_idx]
            if item not in old_state:  # New item found
                returned_data.append(item)
                if mode == "incremental":
                    self.state.append(item)
            else:
                old_state.remove(item)  # Remove to handle duplicates correctly
        
        if mode == "diff":  # Also add removed items
            returned_data.extend(old_state)
            self.state = new_state
        
        # Enforce maximum entries
        if len(self.state) > self.maximum_entries:
            _ = self.state.pop(0)
        await self.write(self.state)
        return returned_data
        
                