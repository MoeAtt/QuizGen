import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain

os.environ['OPENAI_API_KEY'] = apikey

llm = OpenAI(openai_api_key=apikey, temperature=0.8 )

title_template = PromptTemplate(
    input_variables=["topic","num_questions"],
    template='Write me an exam about {topic} make it {num_questions} question with Multiple choice a,b,c,d'
)
answers_chain = LLMChain(
    llm=llm,
    prompt = title_template,
    verbose = True
)


# UI

st.title('PwC')
prompt = st.text_input('Please Write your topic for Exam generator')


num_questions_textbox = st.text_input('Enter the number of questions', type="default")
if num_questions_textbox:
    num_questions = int(num_questions_textbox)
    generate_exam = True
else:
    generate_exam = False
# Exam Generation


if st.button("Generate Exam") and prompt and num_questions_textbox:
    with st.spinner("Generating..."):
        # Generate exam questions
        output = answers_chain.run(topic=prompt, num_questions=num_questions)
        questions_and_answers = output.split('\n\n')

        # Display questions and create radio buttons for each answer
        for i, question_and_answer in enumerate(questions_and_answers):
            parts = question_and_answer.split('\n', 1)

            if len(parts) == 2:
                question, options = parts
                options = options.split('\n')

                st.write(f"Q{i + 1}. {question}")

                # Create radio buttons for each answer
                selected_option = st.radio(f"Select an answer for Q{i + 1}:", options)

                st.write(f"You selected: {selected_option}")
            else:
                st.warning(f"Unexpected format for question {i + 1}: {question_and_answer}")