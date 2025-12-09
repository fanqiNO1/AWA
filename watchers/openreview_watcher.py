"""
OpenReview Watcher - Monitors OpenReview submissions for updates
Sends notifications for new reviews, comments, and status changes

Version 0.2.0 - Integrated with StateDiff plugin for state management
"""

import asyncio
import hashlib
from typing import Any
from collections.abc import Coroutine

import openreview
from loguru import logger

from notifier import Notifier
from plugins.llm import query
from plugins.state_diff import StateDiff, JsonSerializable

BASE_URL = "https://api2.openreview.net"
DEFAULT_INTERVAL_SECONDS = 120


class OpenReviewClientNeedsReplacement(Exception):
    """
    Exception raised when OpenReview client encounters auth/rate limit errors
    and needs to be replaced with a fresh instance.
    """
    pass


class OpenReviewWatcherClient(openreview.api.OpenReviewClient):
    """Extended OpenReview client with watcher-specific functionality."""

    def __init__(
        self,
        username: str,
        password: str,
        user_id: str,
        summary_model: str = "gpt-4o-mini",
        config_name: str = "",
    ):
        """
        Initialize the OpenReview watcher client.

        Args:
            username: OpenReview username
            password: OpenReview password
            user_id: OpenReview user ID (e.g., "~John_Doe1")
            summary_model: LLM model name for summarization
            config_name: Config file name for user-specific caching
        """
        super().__init__(baseurl=BASE_URL, username=username, password=password)
        self._watcher_user_id = user_id
        self._summary_model = summary_model
        self._config_name = config_name

        # StateDiff instances per forum for persistent state tracking
        self._forum_state_diffs: dict[str, StateDiff] = {}

        # Create a sanitized user_id for use in file names
        # Include config_name to separate caches across different config files
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:12]
        self._safe_user_id = f"{config_name}-{user_hash}" if config_name else user_hash

    def _get_forum_state_diff(self, forum_id: str) -> StateDiff:
        """
        Get or create a StateDiff instance for a forum.

        Args:
            forum_id: OpenReview forum ID

        Returns:
            StateDiff instance for this forum
        """
        if forum_id not in self._forum_state_diffs:
            # Create unique identifier for this forum
            forum_hash = hashlib.md5(forum_id.encode()).hexdigest()[:12]
            self._forum_state_diffs[forum_id] = StateDiff(
                app_id="openreview_watcher",
                user_id=f"{self._safe_user_id}_{forum_hash}",
                maximum_entries=1000  # Keep up to 1000 notes per forum
            )
        return self._forum_state_diffs[forum_id]

    async def list_my_submissions(self) -> list[openreview.api.client.Note]:
        """Retrieve all papers associated with the watcher user."""
        all_notes = []
        offset = 0
        limit = 50  # Fetch in batches

        while True:
            try:
                # Use asyncio.to_thread for blocking API call
                batch = await asyncio.to_thread(
                    self.get_notes,
                    content={"authorids": self._watcher_user_id},
                    limit=limit,
                    offset=offset,
                )

                if not batch:
                    break

                all_notes.extend(batch)
                logger.debug(f"Fetched {len(batch)} notes (offset={offset})")

                if len(batch) < limit:
                    break

                offset += limit

            except Exception as e:
                logger.error(f"Error fetching submissions at offset {offset}: {e}")
                break

        return all_notes

    async def list_my_active_submissions(self) -> list[openreview.api.client.Note]:
        """Retrieve active papers associated with the watcher user."""
        all_notes = await self.list_my_submissions()

        # Filter for active submissions (those without an outcome date)
        active_notes = []
        for note in all_notes:
            if note.odate is not None:
                active_notes.append(note)

        return active_notes

    async def get_forum_details(
        self, forum_id: str
    ) -> list[openreview.api.client.Note]:
        """Retrieve all notes (reviews/comments) for a forum (paper)."""
        all_reviews = []
        offset = 0
        limit = 100

        while True:
            try:
                batch = await asyncio.to_thread(
                    self.get_notes,
                    forum=forum_id,
                    sort="tmdate:asc",
                    limit=limit,
                    offset=offset,
                )

                if not batch:
                    break

                all_reviews.extend(batch)

                if len(batch) < limit:
                    break

                offset += limit

            except Exception as e:
                logger.error(f"Error fetching forum details at offset {offset}: {e}")
                break

        logger.debug(f"Retrieved {len(all_reviews)} notes for forum {forum_id}")
        return all_reviews

    async def note_to_markdown(
        self, note: openreview.api.client.Note, is_summary: bool = False
    ) -> str:
        """
        Convert a note (review/comment) to markdown format.

        First extracts key fields (title, pdf, rating, confidence, recommendation, decision),
        then sends remaining content to LLM for summarization.

        Args:
            note: The OpenReview note to convert
            is_summary: If True, generate a summary for remaining content
        """
        md = ""

        # Helper function to extract value from field
        def extract_value(field_data):
            if isinstance(field_data, dict) and "value" in field_data:
                return field_data["value"]
            return field_data

        # 1. Handle title
        if "title" in note.content:
            title = extract_value(note.content.pop("title"))
            md += f"## {title}\n\n"

        # 2. Handle PDF and supplementary material links
        if "pdf" in note.content:
            pdf_value = extract_value(note.content.pop("pdf"))
            md += f"[📄 PDF]({BASE_URL}{pdf_value})"

            if "supplementary_material" in note.content:
                supp_value = extract_value(note.content.pop("supplementary_material"))
                md += f" | [📎 Supplementary Material]({BASE_URL}{supp_value})"
            md += "\n\n"

        # 3. Handle rating
        if "rating" in note.content:
            rating = extract_value(note.content.pop("rating"))
            md += f"- 💯 **Rating:** {rating}\n"

        # 4. Handle confidence
        if "confidence" in note.content:
            confidence = extract_value(note.content.pop("confidence"))
            md += f"- 🔒 **Confidence:** {confidence}\n"

        # 5. Handle recommendation
        if "recommendation" in note.content:
            recommendation = extract_value(note.content.pop("recommendation"))
            md += f"- 📝 **Recommendation:** {recommendation}\n"

        # 6. Handle decision
        if "decision" in note.content:
            decision = extract_value(note.content.pop("decision"))
            md += f"- ⚖️ **Decision:** {decision}\n"

        # 7. Everything else goes to LLM for summarization
        if note.content and is_summary:
            md += "\n"
            try:
                # Dump remaining content as JSON
                import json
                remaining_content = {}
                for key, value in note.content.items():
                    remaining_content[key] = extract_value(value)

                content_json = json.dumps(remaining_content, indent=2, ensure_ascii=False)

                # Send to LLM for summarization
                summary = await query(
                    message=content_json,
                    model=self._summary_model,
                    system_message="You are an expert research paper reviewer. The user will provide you with a JSON containing review/comment data. Summarize the key points concisely in no more than 3 sentences. Focus on the most important information.",
                )
                md += f"{summary}\n*(summarized)*\n\n"
            except Exception as e:
                logger.error(f"Failed to summarize remaining content: {e}", exc_info=True)
                # Fallback: just list the keys
                md += f"**Additional content:** {', '.join(note.content.keys())}\n\n"
        elif note.content:
            # If not summarizing, just list the keys
            md += f"\n**Additional content:** {', '.join(note.content.keys())}\n\n"

        md += f"\n[View on OpenReview]({BASE_URL}/forum?id={note.forum})\n"

        return md.strip()

    def _extract_note_data(self, note: openreview.api.client.Note) -> dict[str, str | int]:
        """
        Extract cacheable data from a note for state tracking.

        Returns a dict compatible with JsonSerializable containing note ID and content hash.
        """
        content_hash = hash(str(note.content))
        return {
            "id": str(note.id),
            "content_hash": content_hash,
        }

    async def tick(self) -> list[str]:
        """
        Check all active submissions for updates and return notifications.

        Returns:
            List of markdown-formatted notification strings

        Raises:
            OpenReviewClientNeedsReplacement: When API calls fail and client needs recreation
        """
        notifications = []

        # Fetch active submissions with error handling
        # If this fails, we need to replace the client
        try:
            active_submissions: list[openreview.api.client.Note] = []
            active_submissions = await self.list_my_active_submissions()
        except Exception as e:
            logger.error(
                f"Failed to fetch active submissions for {self._watcher_user_id}: {e}",
                exc_info=True
            )
            # Signal that this client needs to be replaced
            raise OpenReviewClientNeedsReplacement(
                f"Client for {self._watcher_user_id} failed to fetch submissions: {e}"
            ) from e

        try:
            if not active_submissions:
                return notifications

            # Check each active submission for updates
            for submission in active_submissions:
                forum_id = submission.forum or submission.id

                # Get all notes in this forum
                forum_notes = await self.get_forum_details(forum_id)

                # Get StateDiff instance for this forum
                state_diff = self._get_forum_state_diff(forum_id)

                # Extract current note data
                current_notes: list[JsonSerializable] = [
                    self._extract_note_data(note) for note in forum_notes
                ]

                # Use StateDiff with "diff" mode to detect all changes
                # This returns both new/updated entries and removed entries
                changed_notes_data = await state_diff.diff(
                    current_notes, mode="diff"
                )

                # Build sets for efficient lookup
                current_note_ids = {str(note.id) for note in forum_notes}
                old_note_ids = {
                    str(item["id"]) for item in state_diff.state
                    if isinstance(item, dict) and "id" in item
                } if state_diff.state else set()

                # Separate into new notes, updated notes, and removed notes
                # Entries in diff that are in current_notes are either new or updated
                # Entries in diff that are NOT in current_notes are removed
                new_notes = []
                updated_notes = []
                removed_note_ids = []

                # Create a mapping of note_id to note object
                note_map = {str(note.id): note for note in forum_notes}

                for changed_data in changed_notes_data:
                    if isinstance(changed_data, dict) and "id" in changed_data:
                        note_id = str(changed_data["id"])

                        # Check if this note is in the current state
                        if note_id in current_note_ids:
                            note = note_map.get(note_id)
                            if note:
                                if note_id in old_note_ids:
                                    # Note ID existed before, so it's an update
                                    updated_notes.append(note)
                                else:
                                    # Completely new note
                                    new_notes.append(note)
                        else:
                            # Note is in diff but not in current_notes → it was removed
                            removed_note_ids.append(note_id)
                            logger.info(f"Note {note_id} was removed from forum {forum_id}")

                # Generate notification if there are updates
                if new_notes or updated_notes or removed_note_ids:
                    notification = await self._format_update_notification(
                        submission, new_notes, updated_notes, removed_note_ids
                    )
                    if notification:
                        notifications.append(notification)

        except Exception as e:
            logger.error(f"Error in tick: {e}", exc_info=True)

        return notifications

    async def _format_update_notification(
        self,
        submission: openreview.api.client.Note,
        new_notes: list,
        updated_notes: list,
        removed_note_ids: list[str] | None = None,
    ) -> str | None:
        """Format update notification for a submission."""
        if not new_notes and not updated_notes and not removed_note_ids:
            return None

        if removed_note_ids is None:
            removed_note_ids = []

        # Get submission title
        title = "Unknown Paper"
        if "title" in submission.content:
            title_content = submission.content["title"]
            if isinstance(title_content, dict) and "value" in title_content:
                title = title_content["value"]
            else:
                title = title_content

        # Build notification
        md = "# 📝 OpenReview Update\n\n"
        md += f"**Paper:** [{title}]({BASE_URL}/forum?id={submission.forum or submission.id})\n\n"

        # Add new notes
        if new_notes:
            md += f"## 🆕 New Activity ({len(new_notes)} items)\n\n"
            for note in new_notes[:5]:  # Limit to 5 most recent
                note_type = self._get_note_type(note)
                md += f"### {note_type}\n"
                note_md = await self.note_to_markdown(note, is_summary=True)
                md += note_md + "\n\n---\n\n"

            if len(new_notes) > 5:
                md += f"*...and {len(new_notes) - 5} more new items*\n\n"

        # Add updated notes (brief mention)
        if updated_notes:
            md += f"## 📝 Updates ({len(updated_notes)} items)\n\n"
            for note in updated_notes[:3]:
                note_type = self._get_note_type(note)
                md += f"- {note_type} was updated\n"

            if len(updated_notes) > 3:
                md += f"*...and {len(updated_notes) - 3} more updates*\n\n"

        # Add removed notes
        if removed_note_ids:
            md += f"## 🗑️ Removed ({len(removed_note_ids)} items)\n\n"
            for note_id in removed_note_ids[:5]:
                md += f"- Note `{note_id}` was removed\n"

            if len(removed_note_ids) > 5:
                md += f"*...and {len(removed_note_ids) - 5} more removals*\n"

        return md

    def _get_note_type(self, note: openreview.api.client.Note) -> str:
        """Determine the type of a note based on its signatures."""
        signatures = note.signatures if note.signatures else []

        for sig in signatures:
            sig_lower = sig.lower()
            if "review" in sig_lower:
                return "📊 Review"
            elif "comment" in sig_lower or "rebuttal" in sig_lower:
                return "💬 Comment"
            elif "decision" in sig_lower:
                return "⚖️ Decision"
            elif "meta" in sig_lower:
                return "📋 Meta Review"

        return "📝 Note"


class OpenReviewMonitor:
    """Monitor for OpenReview submissions."""

    def __init__(self, notifier: Notifier, config: dict):
        """
        Initialize the OpenReview monitor.

        Args:
            notifier: Notifier instance for sending alerts
            config: Configuration dictionary
        """
        self.notifier = notifier
        self.config = config
        self.interval = config.get("interval_seconds", DEFAULT_INTERVAL_SECONDS)

        # Get config_name from injected config name (for user-specific caching)
        # This is automatically injected by main.py from the config filename
        self._config_name = str(config.get("_config_name", ""))

        # Store client configurations for recreation
        self.client_configs: list[dict] = config.get("clients", [])

        # Initialize clients for each configured user
        self.clients: list[OpenReviewWatcherClient] = []
        self._initialize_clients()

        # Track if this is the first run
        self.first_run = True

    def _initialize_clients(self) -> None:
        """Initialize OpenReview clients from configurations."""
        for client_config in self.client_configs:
            try:
                # Get summary model from config, fallback to default
                summary_model = client_config.get("summary_model", "gpt-4o-mini")

                client = OpenReviewWatcherClient(
                    username=client_config["username"],
                    password=client_config["password"],
                    user_id=client_config["user_id"],
                    summary_model=summary_model,
                    config_name=self._config_name,
                )
                self.clients.append(client)
                logger.info(
                    f"Added OpenReview client for user: {client_config['user_id']}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize client for {client_config.get('username')}: {e}"
                )

    def _recreate_client(self, failed_client: OpenReviewWatcherClient) -> OpenReviewWatcherClient | None:
        """
        Recreate a failed OpenReview client with fresh authentication.

        Args:
            failed_client: The client that failed and needs replacement

        Returns:
            New client instance or None if recreation failed
        """
        # Find the config for this client
        failed_user_id = failed_client._watcher_user_id

        for client_config in self.client_configs:
            if client_config["user_id"] == failed_user_id:
                try:
                    logger.info(f"Recreating OpenReview client for {failed_user_id}")

                    # Get summary model from config, fallback to default
                    summary_model = client_config.get("summary_model", "gpt-4o-mini")

                    new_client = OpenReviewWatcherClient(
                        username=client_config["username"],
                        password=client_config["password"],
                        user_id=client_config["user_id"],
                        summary_model=summary_model,
                        config_name=self._config_name,
                    )
                    logger.info(f"Successfully recreated client for {failed_user_id}")
                    return new_client
                except Exception as e:
                    logger.error(
                        f"Failed to recreate client for {failed_user_id}: {e}",
                        exc_info=True
                    )
                    return None

        logger.error(f"Could not find config for failed client {failed_user_id}")
        return None

    async def check_all_clients(self) -> None:
        """Check all configured OpenReview accounts for updates."""
        for idx, client in enumerate(self.clients):
            try:
                notifications = await client.tick()

                # Skip notifications on first run to avoid spam
                if self.first_run:
                    if notifications:
                        logger.info(
                            f"First run, skipping {len(notifications)} notifications for {client._watcher_user_id}"
                        )
                    continue

                # Send notifications
                for notification in notifications:
                    await self.notifier.send(notification)

            except OpenReviewClientNeedsReplacement as e:
                logger.warning(
                    f"Client {client._watcher_user_id} needs replacement: {e}"
                )

                # Immediate re-login: recreate client right away
                new_client = self._recreate_client(client)
                if new_client:
                    self.clients[idx] = new_client
                    logger.info(
                        f"Successfully replaced client for {client._watcher_user_id}"
                    )

                    # Retry the tick immediately with the new client
                    try:
                        logger.info(f"Retrying tick with new client for {new_client._watcher_user_id}")
                        notifications = await new_client.tick()

                        # Skip notifications on first run
                        if not self.first_run:
                            for notification in notifications:
                                await self.notifier.send(notification)
                    except Exception as retry_e:
                        logger.error(
                            f"Failed to retry tick after client replacement: {retry_e}",
                            exc_info=True
                        )
                else:
                    logger.error(
                        f"Failed to replace client for {client._watcher_user_id}, "
                        "will retry on next check"
                    )

            except Exception as e:
                logger.error(
                    f"Error checking client {client._watcher_user_id}: {e}",
                    exc_info=True,
                )

        self.first_run = False

    async def monitor_loop(self) -> None:
        """Main monitoring loop."""
        if not self.clients:
            logger.warning("No OpenReview clients configured")
            return

        logger.info(
            f"Starting OpenReview watcher | "
            f"Interval: {self.interval}s | "
            f"Clients: {len(self.clients)}"
        )

        while True:
            try:
                await self.check_all_clients()
                await asyncio.sleep(self.interval)

            except asyncio.CancelledError:
                logger.info("OpenReview watcher cancelled")
                break
            except Exception as e:
                logger.error(f"Error in OpenReview watcher: {e}", exc_info=True)
                await asyncio.sleep(self.interval)

        logger.info("OpenReview watcher stopped")


async def openreview_watcher(notifier: Notifier, config: dict) -> None:
    """
    Monitor OpenReview submissions for updates.

    Args:
        notifier: Notifier instance for sending alerts
        config: Configuration dictionary
    """
    monitor = OpenReviewMonitor(notifier, config)
    await monitor.monitor_loop()


def init(notifier: Notifier, config: dict) -> Coroutine[Any, Any, None]:
    """
    Initialize the OpenReview watcher.

    Args:
        notifier: Notifier instance for sending notifications
        config: Configuration dictionary for this watcher

    Returns:
        Async coroutine or None if disabled
    """
    if not config:
        logger.warning("OpenReview watcher is not configured")
        return None

    if not config.get("enabled", False):
        logger.info("OpenReview watcher is disabled")
        return None

    return openreview_watcher(notifier, config)
