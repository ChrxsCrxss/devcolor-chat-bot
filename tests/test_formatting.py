from devcolor_rag.formatting import plain_terminal_text


def test_strip_bold():
    assert plain_terminal_text("• **Mentorship Programs**: Hello") == "• Mentorship Programs: Hello"


def test_strip_italic():
    assert plain_terminal_text("_demo_ text") == "demo text"


def test_strip_headers():
    assert plain_terminal_text("## Title\n\nBody") == "Title\n\nBody"
