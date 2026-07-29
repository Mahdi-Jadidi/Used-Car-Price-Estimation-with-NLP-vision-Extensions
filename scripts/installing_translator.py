from deep_translator import GoogleTranslator

text = "سلام، حالت چطوره؟"

translation = GoogleTranslator(
    source="fa",
    target="en"
).translate(text)

print(translation)