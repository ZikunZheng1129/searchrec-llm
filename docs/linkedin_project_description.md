# LinkedIn Project Description

I finished SearchRec-LLM, a local production-style portfolio project for large-scale recommendation concepts, e-commerce search, ranking, text/metadata item representation, and candidate-constrained generative recommendation.

The project includes:

- Synthetic local data pipeline and query generation.
- BM25, TF-IDF dense, FAISS-style NumPy fallback, and hybrid retrieval.
- Popularity, itemCF, matrix factorization, user-history embedding, GRU4Rec, and SASRec.
- Multi-stage ranking with feature, MLP, tiny cross-encoder, and mixed rankers.
- Mock-first LLM query understanding and user profile generation.
- Text/metadata item representation with an optional image interface.
- Candidate-constrained GenRec, output parsing, hallucination checks, and evidence-grounded explanations.
- Standardized validation reports and a final leaderboard.
- FastAPI service and Streamlit dashboard for a local demo.

The most important design choice is grounding GenRec in retrieved/ranked catalog candidates. The LLM layer can help interpret, rerank, generate, and explain, but final recommendations must map back to known item IDs.

This is a synthetic-debug-data project, not a production performance claim. I built it to demonstrate ML system design, recommendation/search fundamentals, evaluation discipline, and safe LLM integration.

Suggested hashtags:

`#MachineLearning` `#RecommendationSystems` `#Search` `#Ranking` `#LLM` `#GenAI` `#FastAPI` `#Streamlit` `#MLOps`
