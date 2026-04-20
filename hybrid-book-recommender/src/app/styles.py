"""
Styles and HTML templates for the Gradio UI.
"""


def get_book_details_html(book: dict) -> str:
    """
    Returns a formatted HTML string for the book details view.

    Args:
        book (dict): Dictionary containing book details (title, authors, description, etc.)

    Returns:
        str: HTML string.
    """
    return f"""
    <div style="background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1); display: flex; gap: 20px; margin-top: 20px;">
        <img src="{book["thumbnail"]}" style="width: 120px; height: 180px; object-fit: cover; border-radius: 5px;">
        <div style="flex: 1;">
            <h3 style="margin: 0 0 5px 0; font-size: 1.5em; color: white;">{book["title"]}</h3>
            <p style="margin: 0 0 10px 0; color: #ccc;">by {book["authors"]}</p>
            <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                <span style="background: #eab308; color: black; padding: 2px 8px; border-radius: 4px;
                            font-weight: bold; font-size: 0.9em;">⭐ Rating: {book["rating"]}</span>
                <span style="background: #6366f1; color: white; padding: 2px 8px; border-radius: 4px;
                            font-weight: bold; font-size: 0.9em;">🎭 Mood: {book["mood_score"]}</span>
            </div>
            <p style="line-height: 1.6; color: #ddd;">{book["description"]}</p>
        </div>
    </div>
    """


def get_chat_book_card_html(book: object) -> str:
    """Returns a compact markdown card for a book recommendation in the chat.

    Args:
        book: A BookRecommendation object with title, authors, rating, mood_score,
              category, and description attributes.

    Returns:
        str: Markdown-formatted string for the chatbot.
    """
    title = getattr(book, "title", "Unknown")
    authors = getattr(book, "authors", "Unknown")
    rating = getattr(book, "rating", "N/A")
    mood = getattr(book, "mood_score", "N/A")
    category = getattr(book, "category", "N/A")
    desc = getattr(book, "description", "No description available.")
    truncated = (desc[:200] + "...") if len(desc) > 200 else desc

    return (
        f"\n**📖 {title}** by *{authors}*\n⭐ {rating} · 🎭 {mood} · 📂 {category}\n> {truncated}\n"
    )
