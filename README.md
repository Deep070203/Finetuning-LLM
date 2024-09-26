# Finetuning-LLM

## Direct-Preference-Optimization
Training Large Language Models (LLMs) on extensive datasets in an unsupervised manner has proven highly effective in creating models capable of a wide range of tasks. These models demonstrate a significant breadth of knowledge and understanding of the world. For most applications, it’s crucial for LLMs to generate text that is contextually consistent and aligned with the intended task and user behavior. This includes developing LLMs that are safe, aligned, and unbiased, or those capable of generating syntactically and functionally correct code, despite the presence of incorrect code in the training data. However, the pre-training process alone does not guarantee specific model behavior. This is where Reinforcement Learning From Human Feedback (RLHF) becomes vital.

RLHF is a technique used to fine-tune LLMs by maximizing a reward function derived from another reward model trained on human feedback from evaluators based on a set of generated samples. This technique is widely used and is considered state-of-the-art. However, RLHF has several drawbacks that limit its effectiveness as a solution.

Direct Preference Optimization (DPO), a newly proposed technique addresses these drawbacks and offers a more robust solution. 

The key insight in Direct Preference Optimization is replacing the complex reward modeling process in RLHF with a simple loss function that directly optimizes for human preferences in closed form. It does this by simply increasing the log probability of the tokens in the human prefered responses, and decreasing the log probability of the tokens in the human disprefered responses, given a preferences dataset, which basically makes the model have an implicit reward function that is directly optimized for human preferences. Through this clever math trick, the process now becomes much simpler and more efficient than RLHF, as it does not require a separate reward model, and it is also more stable, as it does not use other methods like PPO for fine-tuning.

## Chatbot-with-Gradio
The provided code demonstrates the implementation of a chatbot, powered by the LLaMA-7B language model. The chatbot interacts with users through a user-friendly Gradio interface, generating responses based on input conversation history. By breaking down the code into various sections, we've explored how the model is loaded, the chatbot's behavior is defined, and the Gradio interface is set up for user interaction.

The Gradio interface is used to create a user-friendly GUI for interacting with the chatbot. Users can input messages, adjust advanced options such as temperature and sampling techniques, and view the chatbot's responses in real-time.

The GUI displays a chat message box where users can type their messages.
The "Submit" button triggers user input processing and chatbot response generation.
Advanced options like temperature, top-p sampling, top-k sampling, and repetition penalty can be adjusted using sliders.
A disclaimer is included to highlight that the model's outputs may not always be factually accurate.
A privacy policy link is provided for user reference.

It's important to note that this demo showcases the capabilities of the LLaMA-7B model and its interaction with users. However, as with any AI-generated content, users should be aware that the model's responses may not always be entirely accurate or appropriate, and they should exercise caution when using the chatbot for factual information or important decisions.


## RAG with Langchain
The provided notebook demonstrates a Retrieval Augmented Generation agent using Langchain, Weaviate and Ollama (Llama 3.1). Use a PDF link and get the data. Transform the data into chunks using Character Text Splitter and store the chunks to vector-store from Weaviate. 

Create a Prompt template to make our model understand what we plan to achieve. Use langchain libraries to store and run the prompt.

Create ChatOllama llm and pass in the prompt, context/pdf and your question. The agent will answer your question using the PDF increasing the accuracy.

## PEFT-Finetuning Qwen2.5
The provided Colab notebook demonstrates PEFT Finetuning of Qwen2.5-1.5B model using the method Low Rank Adaptation "LoRA". First we load the model from Hugging Face, Post-process the model to freeze some parameters.

Then we apply LoRA with the LoRA Config and we use a dataset from Huggingface to train the model.

Then, we save the new LoRA model to our Huggingface account and check if we can load from our account and run it for inference.
