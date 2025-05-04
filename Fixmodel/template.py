class Template:
    def promptSentimentAnalysis(text):
        prompt = f"""Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
        Provide the answer in JSON format with keys: "sentiment", "confidence" (0-1), and "explanation".

        Text: "{text}"

        Answer:"""
        
        return prompt
    
    def promptStory(genre,title):
        prompt = f"""Write a {genre} story titled "{title}"."""
        return prompt