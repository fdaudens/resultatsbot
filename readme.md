# AI Bot Dives Into Quebec Election Stats

A robot using generative artificial intelligence that can answer questions about the recent Quebec election results. Go ahead, give it a try, ask it anything, challenge it! After each response, click on 👍 or 👎. The program records all the questions asked and the generated responses (anonymously), allowing me to refine the AI.

The goal of this exercise is to test the AI's ability to play with numerical data. It may sound simple, but most examples of AI usage involve words and don't follow the same logic. The prospect of interacting with data in a mature language is incredibly enticing, and I wanted to take on this challenge.

For those interested, I've tested several solutions.

The final configuration relies on a Pandas agent from Langchain, GPT-4 as the language model, Langsmith for interaction tracking, and Streamlit for deployment.

I've also restructured the metadata a bit. The clearer they are, the better the agent performs.

It's truly mind-blowing to witness the agent's reasoning abilities to 1) find a method to answer the question and 2) apply it in its interactions with Pandas.

The idea of evaluating the results was inspired by [a similar exercise by Langchain](https://blog.langchain.dev/benchmarking-question-answering-over-csv-data/) using Titanic data.

Other solutions yielded less impressive results. GPT 3.5 Turbo performs decently but has the advantage of being much cheaper: $0.006 compared to approximately $0.12 per interaction with GPT-4.

I also tried PandasAI, which provides interesting results, but integrating it with Langchain proved a bit challenging.

Your interactions will help me refine the "prompts" (commands?) sent to the model. I'm also eager to experiment with fine-tuning GPT 3.5 to see if it brings us closer to GPT-4.

Among other possible improvements:
- Integration of results from the past 10 years, including partial results
- Automatic generation of graphs