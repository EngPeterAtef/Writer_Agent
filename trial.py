from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from utils import (
    count_words_with_bullet_points,
)
template = """You are an experienced writer and author and you will write a tweets in long form sentences using correct English grammar, where the quality would be suitable for an established online publisher.
            create a tweet about {topic} with {wordCount} words
            """

prompt = PromptTemplate(template=template, input_variables=["topic","wordCount"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

MODEL_PATH= os.getenv("MODEL_PATH", "./models/llama-2-7b-chat.gguf.q2_k.bin")

n_gpu_layers = 1 
n_batch = 512

myTopic = 'Mo Salah'
myWordCount = 200
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    temperature=0.7,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

out = llm_chain.run(topic=myTopic, wordCount=myWordCount)
print(out)
print("the number of words with bullet points is: ", count_words_with_bullet_points(out))