from google import genai
import os
from google.genai import types
import gradio as gr

client = genai.Client(api_key=os.getenv('GEMINI_API'))

personalities = {
  "Friendly": """You are a friendly, enthusiastic, and highly encouraging Study Assistant.
  Your goal is to break down complex concepts into simple, beginner-friendly explanations.
  Use analogies and real-world examples that beginners can relate to.
  Always ask a follow-up question to check understanding.""",
  "Academic": """You are a strictly academic, highly detailed, and professional university Professor.
  Use precise, formal terminology, cite key concepts and structure your response.
  Your goal is to break down complex concepts into simple, beginner-friendly explanations.
  Use analogies and real-world examples that beginners can relate to.
  The response should be elaborate as well as concise. Strike a judicious balance.
  Always ask a follow-up question to check understanding."""
}

def study_assistant(question, persona):
  # Updated model name to gemini-2.5-flash
  system_prompt=personalities[persona]
  response = client.models.generate_content(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=1000),

      contents=question
  )
  return response.text

  demo = gr.Interface(
    fn=study_assistant,
    inputs=[gr.Textbox(label="Question", lines=4, placeholder="Ask a question"),
            gr.Radio(choices= list(personalities.keys()), value= "Friendly", label="Personality")],
    outputs=gr.Textbox(label="Explanation", lines=10),
    title = "Study Assistant",
    description="Ask a question to get simple explanation from AI along with analogies and real world examples."

)

  if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000"))
    )
