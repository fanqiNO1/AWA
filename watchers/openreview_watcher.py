"""
OpenReview Watcher - Monitors OpenReview submissions for updates
Sends notifications for new reviews, comments, and status changes

Version 0.1.0 (c) Kunologist & Claude Opus 4.1
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine, Optional, Dict, Set

import openreview
from loguru import logger

from notifier import Notifier
from plugins.llm import query

BASE_URL = "https://api2.openreview.net"
DEFAULT_INTERVAL_SECONDS = 120
DEFAULT_STATE_CACHE_DIR = "./cache/openreview_watcher"


class OpenReviewWatcherClient(openreview.api.OpenReviewClient):
    """Extended OpenReview client with watcher-specific functionality."""

    def __init__(
        self,
        username: str,
        password: str,
        user_id: str,
        cache_dir: str = DEFAULT_STATE_CACHE_DIR,
    ):
        """
        Initialize the OpenReview watcher client.

        Args:
            username: OpenReview username
            password: OpenReview password
            user_id: OpenReview user ID (e.g., "~John_Doe1")
            cache_dir: Directory to store state cache
        """
        super().__init__(baseurl=BASE_URL, username=username, password=password)
        self._watcher_user_id = user_id
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path specific to this user
        safe_user_id = user_id.replace("~", "").replace("/", "_")
        self._cache_file = self._cache_dir / f"{safe_user_id}_state.json"

        # Load cached state
        self._state_cache = self._load_cache()

        # Track current session's seen items
        self._current_session_notes: Dict[str, Dict[str, Any]] = {}

    def _load_cache(self) -> Dict[str, Any]:
        """Load cached state from file."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r") as f:
                    cache = json.load(f)
                    logger.info(f"Loaded cache from {self._cache_file}")
                    return cache
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

        return {
            "forums": {},  # forum_id -> {"last_update": timestamp, "notes": {note_id: note_data}}
            "last_check": None,
        }

    def _save_cache(self) -> None:
        """Save current state to cache file."""
        try:
            # Update with current session data
            for forum_id, notes in self._current_session_notes.items():
                if forum_id not in self._state_cache["forums"]:
                    self._state_cache["forums"][forum_id] = {
                        "last_update": datetime.now().isoformat(),
                        "notes": {},
                    }

                forum_cache = self._state_cache["forums"][forum_id]
                forum_cache["notes"].update(notes)
                forum_cache["last_update"] = datetime.now().isoformat()

            self._state_cache["last_check"] = datetime.now().isoformat()

            with open(self._cache_file, "w") as f:
                json.dump(self._state_cache, f, indent=2)
            logger.debug(f"Saved cache to {self._cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

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

        logger.info(
            f"Retrieved {len(all_notes)} total submissions for user {self._watcher_user_id}"
        )
        return all_notes

    async def list_my_active_submissions(self) -> list[openreview.api.client.Note]:
        """Retrieve active papers associated with the watcher user."""
        all_notes = await self.list_my_submissions()

        # Filter for active submissions (those without an outcome date)
        active_notes = []
        for note in all_notes:
            if note.odate is not None:
                active_notes.append(note)

        logger.info(
            f"Found {len(active_notes)} active submissions out of {len(all_notes)} total"
        )
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

        Args:
            note: The OpenReview note to convert
            is_summary: If True, generate a summary for long content
        """
        md = ""

        # Handle title
        if "title" in note.content:
            title = note.content["title"]
            if isinstance(title, dict) and "value" in title:
                title = title["value"]
            md += f"## {title}\n\n"

        # Handle PDF and supplementary material links
        if "pdf" in note.content:
            pdf_value = note.content["pdf"]
            if isinstance(pdf_value, dict) and "value" in pdf_value:
                pdf_value = pdf_value["value"]
            md += f"[📄 PDF]({BASE_URL}{pdf_value})"

            if "supplementary_material" in note.content:
                supp_value = note.content["supplementary_material"]
                if isinstance(supp_value, dict) and "value" in supp_value:
                    supp_value = supp_value["value"]
                md += f" | [📎 Supplementary Material]({BASE_URL}{supp_value})"
            md += "\n\n"

        # Handle review content
        if "review" in note.content:
            review_text = note.content["review"]
            if isinstance(review_text, dict) and "value" in review_text:
                review_text = review_text["value"]

            if is_summary and len(review_text) > 2500:
                # Summarize long reviews using LLM
                try:
                    summary = await query(
                        message=review_text,
                        model="doubao-seed-1.6",
                        system_message="You are an expert research paper reviewer. Summarize the key points of the review concisely to strictly fewer than 3 sentences.",
                    )
                    md += f"{summary}\n*(summarized)*\n\n"
                except Exception as e:
                    logger.error(f"Failed to summarize review: {e}")
                    # Fallback to truncation
                    md += review_text[:500] + "...\n*(truncated)*\n\n"
            else:
                md += review_text + "\n\n"

        # Handle comment content
        if "comment" in note.content:
            comment_text = note.content["comment"]
            if isinstance(comment_text, dict) and "value" in comment_text:
                comment_text = comment_text["value"]
            md += f"**Comment:** {comment_text}\n\n"

        # Handle ratings and confidence
        if "rating" in note.content:
            rating = note.content["rating"]
            if isinstance(rating, dict) and "value" in rating:
                rating = rating["value"]
            md += f"- 💯 **Rating:** {rating}\n"

        if "confidence" in note.content:
            confidence = note.content["confidence"]
            if isinstance(confidence, dict) and "value" in confidence:
                confidence = confidence["value"]
            md += f"- 🔒 **Confidence:** {confidence}\n"

        if "recommendation" in note.content:
            recommendation = note.content["recommendation"]
            if isinstance(recommendation, dict) and "value" in recommendation:
                recommendation = recommendation["value"]
            md += f"- 📝 **Recommendation:** {recommendation}\n"

        return md.strip()

    def _extract_note_data(self, note: openreview.api.client.Note) -> Dict[str, Any]:
        """Extract cacheable data from a note (before markdown conversion)."""
        return {
            "id": note.id,
            "forum": note.forum,
            "tmdate": note.tmdate,
            "signatures": note.signatures,
            "content_hash": hash(
                str(note.content)
            ),  # Simple hash to detect content changes
        }

    async def tick(self) -> list[str]:
        """
        Check all active submissions for updates and return notifications.

        Returns:
            List of markdown-formatted notification strings
        """
        notifications = []

        try:
            # Get active submissions
            active_submissions = await self.list_my_active_submissions()

            if not active_submissions:
                logger.debug("No active submissions found")
                return notifications

            # Check each active submission for updates
            for submission in active_submissions:
                forum_id = submission.forum or submission.id

                # Get all notes in this forum
                forum_notes = await self.get_forum_details(forum_id)

                # Track notes for this forum in current session
                if forum_id not in self._current_session_notes:
                    self._current_session_notes[forum_id] = {}

                # Check for new or updated notes
                new_notes = []
                updated_notes = []

                cached_forum = self._state_cache["forums"].get(forum_id, {})
                cached_notes = cached_forum.get("notes", {})

                for note in forum_notes:
                    note_data = self._extract_note_data(note)
                    note_id = note_data["id"]

                    # Store in current session
                    self._current_session_notes[forum_id][note_id] = note_data

                    if note_id not in cached_notes:
                        # New note
                        new_notes.append(note)
                    elif (
                        cached_notes[note_id]["content_hash"]
                        != note_data["content_hash"]
                    ):
                        # Updated note
                        updated_notes.append(note)

                # Generate notification if there are updates
                if new_notes or updated_notes:
                    notification = await self._format_update_notification(
                        submission, new_notes, updated_notes
                    )
                    if notification:
                        notifications.append(notification)

            # Save cache after processing all forums
            self._save_cache()

        except Exception as e:
            logger.error(f"Error in tick: {e}", exc_info=True)

        return notifications

    async def _format_update_notification(
        self,
        submission: openreview.api.client.Note,
        new_notes: list,
        updated_notes: list,
    ) -> Optional[str]:
        """Format update notification for a submission."""
        if not new_notes and not updated_notes:
            return None

        # Get submission title
        title = "Unknown Paper"
        if "title" in submission.content:
            title_content = submission.content["title"]
            if isinstance(title_content, dict) and "value" in title_content:
                title = title_content["value"]
            else:
                title = title_content

        # Build notification
        md = f"# 📝 OpenReview Update\n\n"
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
                md += f"*...and {len(updated_notes) - 3} more updates*\n"

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

        # Initialize clients for each configured user
        self.clients: list[OpenReviewWatcherClient] = []

        cache_dir = config.get("state_cache_dir", DEFAULT_STATE_CACHE_DIR)

        for client_config in config.get("clients", []):
            try:
                client = OpenReviewWatcherClient(
                    username=client_config["username"],
                    password=client_config["password"],
                    user_id=client_config["user_id"],
                    cache_dir=cache_dir,
                )
                self.clients.append(client)
                logger.info(
                    f"Added OpenReview client for user: {client_config['user_id']}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize client for {client_config.get('username')}: {e}"
                )

        # Track if this is the first run
        self.first_run = True

    async def check_all_clients(self) -> None:
        """Check all configured OpenReview accounts for updates."""
        for client in self.clients:
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


def init(notifier: Notifier, config: dict) -> Optional[Coroutine[Any, Any, None]]:
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
