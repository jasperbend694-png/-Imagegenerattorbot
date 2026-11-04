import os
import logging
import asyncio
import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
from functools import wraps

import fal_client
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

# ── Configuration ──────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN")
FAL_KEY = os.environ.get("FAL_KEY")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0"))  # Telegram numeric ID
DB_PATH = os.environ.get("DB_PATH", "bot_data.db")

if not BOT_TOKEN or not FAL_KEY:
    raise ValueError("BOT_TOKEN and FAL_KEY must be set")

fal_client.api_key = FAL_KEY

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Database ───────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            user_id INTEGER,
            timestamp REAL
        )
    """)
    conn.commit()
    conn.close()

def db_register_user(user_id, username, first_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (user_id, username, first_name) VALUES (?, ?, ?)",
              (user_id, username or "", first_name or ""))
    conn.commit()
    conn.close()

def db_log_generation(user_id, prompt, model="flux-schnell"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO generations (user_id, prompt, model) VALUES (?, ?, ?)",
              (user_id, prompt, model))
    conn.commit()
    conn.close()

def db_get_history(user_id, limit=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT prompt, created_at FROM generations WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return rows

def db_total_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    n = c.fetchone()[0]
    conn.close()
    return n

def db_total_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM generations")
    n = c.fetchone()[0]
    conn.close()
    return n

def db_daily_active():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    since = datetime.utcnow() - timedelta(days=1)
    c.execute("SELECT COUNT(DISTINCT user_id) FROM generations WHERE created_at >= ?", (since,))
    n = c.fetchone()[0]
    conn.close()
    return n

# ── Rate limiting (simple per-user sliding window) ─────────────
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 3      # max generations per window

def check_rate_limit(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = time.time()
    c.execute("DELETE FROM rate_limits WHERE timestamp < ?", (now - RATE_LIMIT_WINDOW,))
    c.execute("SELECT COUNT(*) FROM rate_limits WHERE user_id = ?", (user_id,))
    count = c.fetchone()[0]
    if count >= RATE_LIMIT_MAX:
        conn.close()
        return False
    c.execute("INSERT INTO rate_limits (user_id, timestamp) VALUES (?, ?)", (user_id, now))
    conn.commit()
    conn.close()
    return True

# ── Deduplication (in‑memory) ──────────────────────────────────
active_requests = set()

# ── Helper: generate image via fal.ai ──────────────────────────
async def generate_image(prompt: str, size: str = "landscape_4_3"):
    """Call fal.ai FLUX.1 [schnell] and return the image URL."""
    # Validate prompt
    if not prompt or len(prompt) < 3 or len(prompt) > 1000:
        raise ValueError("Prompt must be between 3 and 1000 characters.")
    
    # Run blocking fal_client call in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt,
                "image_size": size,
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": True,
            },
        ),
    )
    return result["images"][0]["url"]

# ── Menu builders ──────────────────────────────────────────────
def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎨 Generate Image", callback_data="menu_generate")],
        [InlineKeyboardButton("🖼 My Generated Images", callback_data="menu_history")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings")],
        [InlineKeyboardButton("📖 How To Use", callback_data="menu_help")],
        [InlineKeyboardButton("ℹ️ About Bot", callback_data="menu_about")],
    ])

def back_to_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="back_menu")]
    ])

def generate_again_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Generate Another Image", callback_data="menu_generate")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="back_menu")],
    ])

# ── Command handlers ───────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    db_register_user(user.id, user.username, user.first_name)
    await update.message.reply_text(
        f"👋 Welcome, {user.first_name}!\n\n"
        "I can generate AI images from your text descriptions.\n"
        "Tap a button below to get started.",
        reply_markup=main_menu_keyboard(),
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 **How to use this bot**\n\n"
        "1. Tap **🎨 Generate Image** or send `/imagine`.\n"
        "2. Type a detailed description of the image you want.\n"
        "3. Wait a few seconds while the AI creates your image.\n"
        "4. Receive your image and tap **Generate Another** if you like.\n\n"
        "**Tips for better results:**\n"
        "• Be specific (subject, style, lighting, colors).\n"
        "• Include keywords like *photorealistic*, *anime*, *3D render*, *portrait*, *logo*.\n"
        "• Avoid overly short or vague prompts.\n\n"
        "**Commands:**\n"
        "/start – main menu\n"
        "/imagine – generate an image\n"
        "/help – this message\n"
        "/settings – current settings\n"
        "/history – your recent prompts\n"
        "/about – bot info\n"
        "/cancel – cancel current operation"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 **AI Image Generator Bot**\n\n"
        "Powered by **fal.ai** — FLUX.1 [schnell] model.\n\n"
        "This bot is a demonstration project built with Python and "
        "python-telegram-bot. All images are generated in real time; "
        "no placeholder or fake responses are used.\n\n"
        "Developer: @Imagegenerattorbot"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚙️ **Current Settings**\n\n"
        "• Model: FLUX.1 [schnell]\n"
        "• Aspect ratio: Landscape 4:3 (default)\n"
        "• Quality: Standard (4 inference steps)\n"
        "• Safety filter: On\n\n"
        "Aspect‑ratio selection will be available in a future update."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_to_menu_keyboard())

async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    rows = db_get_history(user_id, 5)
    if not rows:
        await update.message.reply_text("🖼 You haven't generated any images yet.")
        return
    lines = ["🖼 **Your last 5 generations:**\n"]
    for i, (prompt, created) in enumerate(rows, 1):
        short = prompt[:80] + ("…" if len(prompt) > 80 else "")
        lines.append(f"{i}. `{short}`\n   _{created}_")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("awaiting_prompt", None)
    await update.message.reply_text("✅ Operation cancelled.", reply_markup=main_menu_keyboard())

async def cmd_imagine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["awaiting_prompt"] = True
    await update.message.reply_text(
        "🎨 **Describe the image you want to generate:**\n\n"
        "Example: _A beautiful futuristic city at sunset, ultra-realistic 4K._\n\n"
        "Send /cancel to abort.",
        parse_mode=ParseMode.MARKDOWN,
    )

# ── Admin commands ─────────────────────────────────────────────
def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != ADMIN_ID:
            await update.message.reply_text("⛔ Unauthorized.")
            return
        return await func(update, context)
    return wrapper

@admin_only
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👑 **Admin Panel**\n\n"
        "/stats – bot statistics\n"
        "/broadcast <message> – send to all users\n"
        "/daily – daily activity summary"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

@admin_only
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total_users = db_total_users()
    total_images = db_total_images()
    daily = db_daily_active()
    text = (
        f"📊 **Bot Statistics**\n\n"
        f"• Total users: {total_users}\n"
        f"• Total images generated: {total_images}\n"
        f"• Active users (last 24h): {daily}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

@admin_only
async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
    message = " ".join(context.args)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id FROM users")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    success = 0
    for uid in users:
        try:
            await context.bot.send_message(uid, f"📢 {message}")
            success += 1
        except Exception:
            pass
    await update.message.reply_text(f"Broadcast sent to {success}/{len(users)} users.")

@admin_only
async def cmd_daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    daily = db_daily_active()
    await update.message.reply_text(f"📈 Active users in the last 24 hours: {daily}")

# ── Prompt handler (text message after /imagine or button) ─────
async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_prompt"):
        return
    context.user_data.pop("awaiting_prompt", None)
    
    user_id = update.effective_user.id
    prompt = update.message.text.strip()
    
    # Validate prompt
    if len(prompt) < 3:
        await update.message.reply_text("❌ Prompt too short. Please describe your image in at least 3 characters.")
        return
    if len(prompt) > 1000:
        await update.message.reply_text("❌ Prompt too long. Maximum 1000 characters.")
        return
    
    # Rate limit
    if not check_rate_limit(user_id):
        await update.message.reply_text(
            f"⏳ Rate limit exceeded. You can generate at most {RATE_LIMIT_MAX} images per minute.\n"
            "Please wait a moment and try again."
        )
        return
    
    # Deduplication
    key = (user_id, hashlib.md5(prompt.encode()).hexdigest())
    if key in active_requests:
        await update.message.reply_text("⏳ A generation with this prompt is already in progress. Please wait.")
        return
    active_requests.add(key)
    
    status_msg = await update.message.reply_text("🎨 Generating your image… please wait.")
    
    try:
        image_url = await asyncio.wait_for(generate_image(prompt), timeout=60.0)
        # Download image bytes to send directly (avoids exposing fal CDN URL)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            image_bytes = resp.content
        
        await status_msg.delete()
        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"✅ Generated for: _{prompt[:200]}_",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=generate_again_keyboard(),
        )
        db_log_generation(user_id, prompt, "flux-schnell")
    except asyncio.TimeoutError:
        await status_msg.edit_text("⏰ Generation timed out. Please try again with a shorter prompt.")
    except ValueError as e:
        await status_msg.edit_text(f"❌ {e}")
    except fal_client.FalClientError as e:
        logger.error(f"fal.ai error: {e}")
        err = str(e).lower()
        if "credit" in err or "balance" in err or "quota" in err:
            await status_msg.edit_text("💳 The bot's image-generation credits are exhausted. Please contact the admin.")
        elif "rate" in err:
            await status_msg.edit_text("⏳ The AI service is busy. Please wait a moment and try again.")
        elif "invalid" in err and "key" in err:
            await status_msg.edit_text("🔑 Service configuration error. Please contact the admin.")
        else:
            await status_msg.edit_text("❌ Image generation failed. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await status_msg.edit_text("❌ An unexpected error occurred. Please try again.")
    finally:
        active_requests.discard(key)

# ── Callback query handler (all inline buttons) ────────────────
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == "back_menu":
        await query.edit_message_text(
            "🏠 **Main Menu**\nChoose an option:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_keyboard(),
        )
    elif data == "menu_generate":
        context.user_data["awaiting_prompt"] = True
        await query.edit_message_text(
            "🎨 **Describe the image you want to generate:**\n\n"
            "Example: _A beautiful futuristic city at sunset, ultra-realistic 4K._\n\n"
            "Send /cancel to abort.",
            parse_mode=ParseMode.MARKDOWN,
        )
    elif data == "menu_history":
        user_id = query.from_user.id
        rows = db_get_history(user_id, 5)
        if not rows:
            text = "🖼 You haven't generated any images yet."
        else:
            lines = ["🖼 **Your last 5 generations:**\n"]
            for i, (prompt, created) in enumerate(rows, 1):
                short = prompt[:80] + ("…" if len(prompt) > 80 else "")
                lines.append(f"{i}. `{short}`\n   _{created}_")
            text = "\n".join(lines)
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_to_menu_keyboard())
    elif data == "menu_settings":
        text = (
            "⚙️ **Current Settings**\n\n"
            "• Model: FLUX.1 [schnell]\n"
            "• Aspect ratio: Landscape 4:3\n"
            "• Quality: Standard\n"
            "• Safety filter: On"
        )
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_to_menu_keyboard())
    elif data == "menu_help":
        text = (
            "📖 **How to use**\n\n"
            "Tap **🎨 Generate Image**, then type your description.\n"
            "The AI will create an image and send it here.\n\n"
            "Use /history to see your recent prompts."
        )
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_to_menu_keyboard())
    elif data == "menu_about":
        text = (
            "🤖 **AI Image Generator Bot**\n\n"
            "Powered by fal.ai — FLUX.1 [schnell].\n"
            "Real AI generation, no placeholders."
        )
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_to_menu_keyboard())

# ── Error handler ──────────────────────────────────────────────
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error: {context.error}")
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ An unexpected error occurred. Please try again or /start."
            )
        except Exception:
            pass

# ── Main ───────────────────────────────────────────────────────
def main():
    init_db()
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("imagine", cmd_imagine))
    
    # Admin
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("broadcast", cmd_broadcast))
    app.add_handler(CommandHandler("daily", cmd_daily))
    
    # Inline buttons
    app.add_handler(CallbackQueryHandler(button_handler))
    
    # Text prompts (only when awaiting)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prompt))
    
    # Error handler
    app.add_error_handler(error_handler)
    
    logger.info("Bot started (polling).")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
