from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
import openai
import json

class ChatbotView(View):
    def get(self, request):
        return render(request, 'chatbot.html')
    

    def get_api_key(self):
        with open('/Users/amrmohamed/Downloads/openai/apikey.txt', 'r') as file:
            return file.read().strip()

    def post(self, request):
        # Set the OpenAI API key
        openai.api_key = self.get_api_key()

        # Read the ID of the fine-tuned model from the file
        with open('/Users/amrmohamed/Downloads/openai/model_id.txt', 'r') as file:
            fine_tuned_model_id = file.read().strip()
            self.fine_tuned_model_id = fine_tuned_model_id

        # Get the user's message from the request
        message = json.loads(request.body)["message"]

        # Generate a response using the fine-tuned model
        response = openai.Completion.create(
            #engine="davinci",
            #model="gpt-3.5-turbo-" + self.fine_tuned_model_id,
            model = self.fine_tuned_model_id,
            prompt=message,
            max_tokens=150  # you can adjust this value as needed
        )

        # Return the response as a JSON object
        return JsonResponse({"response": response.choices[0].text.strip()})
