import gradio as gr

# Create a sample code component to check supported languages
try:
    print("Attempting to list all supported languages in Gradio Code component:")
    
    # Try to access the internal list of supported languages if available
    from gradio.components.code import list_supported_languages
    languages = list_supported_languages()
    
    if languages:
        print("\nSupported languages:")
        for lang in sorted(languages):
            print(f"- {lang}")
    else:
        print("Couldn't retrieve supported languages programmatically.")
        
except (ImportError, AttributeError):
    print("Couldn't access language list directly. Trying a few common ones:")
    
    # Try some common languages to see which ones work
    common_languages = [
        "python", "javascript", "html", "css", "bash", "json", 
        "markdown", "typescript", "clike", "c", "cpp", "java",
        "rust", "go", "ruby", "php", "shell", "sql", "xml"
    ]
    
    print("\nTesting common languages:")
    for lang in common_languages:
        try:
            # Try to create a Code component with this language
            code = gr.Code(language=lang)
            print(f"✓ {lang} is supported")
        except ValueError:
            print(f"✗ {lang} is NOT supported")

print("\nNote: Run this script to check which languages are supported in your version of Gradio.")