class Template:
    def promptSentimentAnalysis(text):
        prompt = f"""Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
        Provide the answer in JSON format with keys: "sentiment" and "confidence" (0-1).

        Text: "{text}"

        Answer:"""
        
        return prompt
    
    def promptStory(genre,title):
        prompt = f"""Write a {genre} story titled "{title}". The story should be at least 500 words long."""
        return prompt
    
    
    def promptStory():
        prompt = f"""
        Write a humorous story in English with:  
        - **Premise**: [e.g., "A zombie who hates brains"]  
        - **Style**: [Sarcastic, Absurd, Satire] """
        
        return prompt
    
    def promptEmail():
        prompt =f"""Generate a professional business email in English with the following details:  
        - Purpose: [Request a meeting]  
        - Tone: [Formal]
        - Word limit: [100-150 words]  

        Example output format:  
        Subject: [Clear and concise subject line]  
        Body: [Well-structured email content] """
        return prompt
    
    def promptSocialMedia():
        prompt =f"""
        Create a engaging social media post in English for Informatic UNPAR about new student.  
        Requirements:  
        - Platform: Twitter
        - Hashtags: #ifUnpar, #informatikaunpar, #informatika
        - Tone: Informative 
        - Length: 1-2 sentences for Twitter 
        """
        return prompt
    
    def promptSortStory():
        prompt=f"""Write a short story in English with the following elements:  
        - **Genre**: Fantasy, Mystery, Romance  
        - **Main character**: "A retired detective with a fear of heights"
        - **Setting**: "A spaceship in the year 2150" 
        - **Conflict**:"The crew disappears one by one"
        - **Ending**: Happy  
        - **Word limit**: 300-500 words  

        Additional instructions:  
        - Use vivid descriptions and dialogue to make the story engaging.
        """
        return prompt
        
    
    def promptBebas(text):
        return text