import requests # Kept for potential robots.txt, though less critical if not scraping
import re
import json
from typing import List, Dict, Any
import time
import random
from urllib.parse import urlparse, urljoin
import google.generativeai as genai
import os
from serpapi import GoogleSearch
# Removed playwright imports
# Removed asyncio imports
from urllib.robotparser import RobotFileParser # Still useful for politeness check, though not strictly required if only using snippets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Configuration ---
# IMPORTANT: DO NOT HARDCODE API KEYS IN PRODUCTION CODE.
# Use environment variables or a secure configuration management system.
GEMINI_API_KEY = "" # Recommended way
# If GEMINI_API_KEY environment variable is not set, uncomment the line below
# and replace "YOUR_DEFAULT_GEMINI_KEY_HERE" with your actual key (for testing only).
# if not GEMINI_API_KEY:
#     GEMINI_API_KEY = "YOUR_DEFAULT_GEMINI_KEY_HERE" # Replace with your key or handle missing key error

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set or hardcoded default missing.")

SERPAPI_API_KEY = "" # Your provided key


genai.configure(api_key=GEMINI_API_KEY)


class SearchTool:
    """Tool for performing searches using SerpApi."""

    def __init__(self, api_key):
        self.api_key = api_key
        logging.info("Initialized SearchTool with SerpApi.")

    def search(self, query: str, num_results: int = 10, include_news: bool = False) -> List[Dict[str, Any]]:
        """Perform Google search using SerpApi."""
        logging.info(f"Searching for: '{query}'")
        params = {
            "q": query,
            "hl": "en",  # Language
            "gl": "us",  # Country (can be adjusted)
            "api_key": self.api_key,
            "num": num_results,
            "safe": "active" # Safe search
        }

        if include_news:
            params["tbm"] = "nws" # Tab for news results

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            organic_results = results.get("organic_results", [])
            news_results = results.get("news_results", [])
            # For news, SerpApi often puts results in 'news_results' AND 'organic_results'
            # Let's combine and ensure uniqueness by link
            combined_results_dict = {}
            for res in organic_results + news_results:
                 if res.get("link"):
                     # Add a flag to easily identify news results later if they came from news_results
                     if res in news_results:
                         res['is_news'] = True
                     combined_results_dict[res["link"]] = res # Use link as key to handle duplicates

            combined_results = list(combined_results_dict.values())


            logging.info(f"Found {len(combined_results)} total search results (organic and news).")
            return combined_results

        except Exception as e:
            logging.error(f"Error during SerpApi search: {e}")
            return []

# Removed the WebScraper class entirely. We will rely on SerpApi snippets.


class ContentAnalyzer:
    """Tool for analyzing and evaluating content relevance and extracting info using AI."""

    def __init__(self):
        """Initialize the content analyzer."""
        logging.info("Initializing ContentAnalyzer.")
        try:
            # Use a slightly more capable model for analysis tasks
            self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini Pro model, falling back to Flash: {e}")
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    def analyze_relevance(self, content: str, query: str) -> float:
        """Use AI to analyze content relevance to the query."""
        if not content or len(content) < 20: # Skip analysis for very short or empty content
             return 0.1 # Assume low relevance if content is minimal (e.g. empty snippet)

        try:
            # Truncate content if it's excessively long to fit within model limits
            # Snippets from SerpApi are usually short, but being safe.
            max_tokens = 10000 # Snippets are much smaller than full pages
            if len(content) > max_tokens * 4: # Estimate token count roughly
                 content = content[:max_tokens * 4] + "..." # Truncate

            prompt = f"""
            Task: Evaluate the relevance of the provided text content to the given search query.
            This content is likely a search result snippet, so focus on whether the snippet
            indicates the linked page contains information relevant to the query.

            Query: "{query}"

            Content (Snippet):
            ```text
            {content}
            ```

            Return ONLY a relevance score from 0.0 to 1.0, where:
            - 0.0 means completely irrelevant (e.g., error snippet, unrelated topic)
            - 0.5 means partially relevant (e.g., mentions the topic but isn't directly about it)
            - 1.0 means highly relevant (snippet strongly suggests the page directly addresses the query)

            Consider the overall content and if it would be helpful for answering the query.
            Just return the numeric score. Do not include any other text or explanation.
            """

            response = self.model.generate_content(prompt)
            score_text = response.text.strip()

            try:
                relevance_score = float(score_text)
                relevance_score = max(0.0, min(relevance_score, 1.0))
                logging.debug(f"Relevance score for query '{query}': {relevance_score}")
                return relevance_score
            except ValueError:
                logging.warning(f"AI returned non-float relevance score: '{score_text}'. Attempting fallback relevance estimation.")
                # Fallback: Estimate relevance based on query term presence
                query_terms = [term.lower() for term in re.findall(r'\b\w+\b', query)] # Extract words
                if not query_terms: return 0.1
                content_lower = content.lower()
                # Count unique query terms found
                found_terms_count = sum(1 for term in set(query_terms) if term in content_lower)
                # Simple score: proportion of unique terms found
                fallback_score = found_terms_count / len(set(query_terms))
                # Adjust score based on content length heuristic (longer content might dilute terms) - very rough
                length_penalty = min(1.0, len(content) / 500) # Arbitrary scaling for snippets
                fallback_score = fallback_score * (1 - length_penalty * 0.1) # Minor penalty for longer snippets

                fallback_score = max(0.0, min(fallback_score, 1.0))
                logging.debug(f"Fallback relevance score: {fallback_score}")
                return fallback_score

        except Exception as e:
            logging.error(f"Error in analyze_relevance AI call: {str(e)}. Falling back to keyword estimation.")
            # Robust Fallback: Keyword presence
            query_terms = [term.lower() for term in re.findall(r'\b\w+\b', query)]
            if not query_terms: return 0.1
            content_lower = content.lower()
            found_terms_count = sum(1 for term in set(query_terms) if term in content_lower)
            fallback_score = found_terms_count / len(set(query_terms))
            fallback_score = max(0.0, min(fallback_score * 0.8, 0.7)) # Cap fallback score as it's less reliable
            logging.debug(f"Robust Fallback relevance score: {fallback_score}")
            return fallback_score


    def extract_key_points(self, content: str, query: str) -> List[str]:
        """Use AI to extract key points from content related to query."""
        if not content or len(content) < 50:
            return ["Snippet too short to extract key points."]

        try:
            # Truncate content similarly to relevance analysis
            max_tokens = 10000
            if len(content) > max_tokens * 4:
                 content = content[:max_tokens * 4] + "..."

            prompt = f"""
            Task: Extract the most important key points from the provided text snippet that are relevant to the given search query.
            Focus on factual information and main ideas directly related to the query, as suggested by the snippet.

            Query: "{query}"

            Content (Snippet):
            ```text
            {content}
            ```

            Return a list of 2-5 concise key points as sentences or short phrases, *based only on the information in the snippet*.
            Format the output as a JSON array of strings.
            Example: ["Snippet mentions X feature", "Source discusses Y aspect", "According to the snippet, Z is happening"]
            If the snippet contains little relevant information, return a list with a single item indicating that.
            """

            response = self.model.generate_content(prompt)

            try:
                # Attempt to parse JSON array
                key_points = json.loads(response.text.strip())
                if isinstance(key_points, list) and all(isinstance(item, str) for item in key_points):
                     # Filter out empty points and clean whitespace
                    cleaned_points = [point.strip() for point in key_points if point.strip()]
                    return cleaned_points if cleaned_points else ["AI extracted no specific key points from snippet."]
                else:
                     logging.warning(f"AI did not return a JSON array of strings for key points: {response.text}")
                     raise ValueError("Unexpected AI output format") # Trigger fallback

            except (json.JSONDecodeError, ValueError):
                 logging.warning(f"AI key point extraction failed to parse or format unexpectedly. Attempting regex fallback.")
                 # Fallback: Extract sentences containing query terms from the snippet
                 sentences = re.split(r'(?<=[.!?])\s+', content)
                 query_terms = [term.lower() for term in re.findall(r'\b\w+\b', query)]

                 fallback_points = []
                 for sentence in sentences:
                     if any(term in sentence.lower() for term in query_terms):
                         fallback_points.append(sentence.strip())
                         if len(fallback_points) >= 3: break # Limit fallback points for snippets

                 return fallback_points if fallback_points else ["No key points extracted by fallback from snippet."]

        except Exception as e:
            logging.error(f"Error in extract_key_points AI call: {str(e)}. Falling back to sentence extraction from snippet.")
            # Robust Fallback: Extract sentences containing query terms
            sentences = re.split(r'(?<=[.!?])\s+', content)
            query_terms = [term.lower() for term in re.findall(r'\b\w+\b', query)]

            fallback_points = []
            for sentence in sentences:
                if any(term in sentence.lower() for term in query_terms):
                    fallback_points.append(sentence.strip())
                    if len(fallback_points) >= 3: break # Limit fallback points for snippets

            return fallback_points if fallback_points else ["No key points extracted by robust fallback from snippet."]


# Removed NewsAggregator

# --- Main Agent Class ---

class WebResearchAgent:
    """Main Web Research Agent that coordinates tools and processes user queries."""

    def __init__(self, serpapi_key: str):
        """Initialize the agent and its tools."""
        self.search_tool = SearchTool(serpapi_key)
        # Removed web_scraper
        self.content_analyzer = ContentAnalyzer()

        # Use Gemini Flash for initial query analysis (faster)
        self.query_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        # Use Gemini Pro for final report synthesis (more capable)
        try:
            self.report_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        except Exception as e:
             logging.error(f"Failed to initialize Gemini Pro for report, falling back to Flash: {e}")
             self.report_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

        logging.info("WebResearchAgent initialized.")

    # Removed async context manager methods (__aenter__, __aexit__)


    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Use AI to analyze the user query to determine research strategy."""
        logging.info(f"Analyzing query: '{query}'")
        try:
            prompt = f"""
            Task: Analyze this research query and provide information to guide a web search.
            Identify the main topic, the user's intent, and if they are looking for recent news.

            Query: "{query}"

            Provide the analysis in JSON format with the following keys:
            - 'original_query': The exact input query.
            - 'query_type': Categorize intent (e.g., 'factual', 'exploratory', 'news', 'comparison', 'how-to'). Choose the best fit.
            - 'search_terms': A refined list of keywords or phrases optimal for a search engine.
            - 'include_news': boolean (true/false) - Should recent news results be prioritized or included?

            Example Output:
            ```json
            {{
              "original_query": "latest news on AI in healthcare",
              "query_type": "news",
              "search_terms": "AI in healthcare news latest developments",
              "include_news": true
            }}
            ```

            ```json
            {{
              "original_query": "what is photosynthesis",
              "query_type": "factual",
              "search_terms": "definition of photosynthesis process",
              "include_news": false
            }}
            ```

            Return ONLY the JSON object.
            """

            response = self.query_model.generate_content(prompt)
            response_text = response.text.strip()

            # Clean potential markdown formatting (```json ... ```)
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(response_text)

                # Validate and sanitize the parsed data
                analysis = {
                    "original_query": query,
                    "query_type": parsed.get("query_type", "exploratory").lower(), # Default and lowercase
                    "search_terms": parsed.get("search_terms", query).strip(), # Default and strip whitespace
                    "include_news": bool(parsed.get("include_news", False)) # Ensure boolean type
                }
                logging.info(f"Query analysis successful: {analysis}")
                return analysis

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logging.warning(f"AI returned malformed JSON or missing keys: {response_text}. Error: {e}. Attempting regex fallback.")
                # Fallback: Use regex to extract components if JSON parsing fails
                query_type_match = re.search(r'"query_type"\s*:\s*"([^"]+)"', response_text)
                search_terms_match = re.search(r'"search_terms"\s*:\s*"([^"]+)"', response_text)
                include_news_match = re.search(r'"include_news"\s*:\s*(true|false)', response_text, re.IGNORECASE)

                fallback_analysis = {
                    "original_query": query,
                    "query_type": query_type_match.group(1).lower() if query_type_match else "exploratory",
                    "search_terms": search_terms_match.group(1).strip() if search_terms_match else query,
                    "include_news": include_news_match.group(1).lower() == "true" if include_news_match else ("news" in query.lower() or random.random() > 0.7) # Heuristic fallback for news
                }
                logging.warning(f"Query analysis fallback result: {fallback_analysis}")
                return fallback_analysis

        except Exception as e:
            logging.error(f"Error in analyze_query AI call: {str(e)}. Falling back to simple keyword analysis.")
            # Robust Fallback: Very basic analysis
            query_lower = query.lower()
            is_news_related = any(term in query_lower for term in ["news", "recent", "latest", "update"])
            is_factual = any(term in query_lower for term in ["what is", "who is", "when did", "where is", "define"])

            query_type = "factual" if is_factual else "news" if is_news_related else "exploratory"

            fallback_analysis = {
                "original_query": query,
                "query_type": query_type,
                "search_terms": query,
                "include_news": is_news_related or random.random() > 0.5 # Random chance to include news
            }
            logging.warning(f"Robust Query analysis fallback result: {fallback_analysis}")
            return fallback_analysis


    def search_web(self, analyzed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform web search based on analyzed query using SerpApi."""
        search_terms = analyzed_query["search_terms"]
        include_news = analyzed_query["include_news"]

        # SerpApi handles both organic and news if include_news is True and results are available
        results = self.search_tool.search(search_terms, include_news=include_news)

        logging.info(f"Received {len(results)} total search results from SerpApi.")
        return results

    def extract_content(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract content (snippets) from search results."""
        extracted_content = []

        logging.info(f"Attempting to extract snippets from {len(search_results)} search results.")
        for i, result in enumerate(search_results):
            url = result.get("link")
            if not url:
                logging.warning(f"Result {i+1} missing URL, skipping extraction.")
                continue

            snippet = result.get("snippet") or result.get("snippet_highlighted_words") # Use snippet or highlighted words

            if snippet:
                 logging.info(f"Found snippet for result {i+1}/{len(search_results)}: {url}")
                 extracted_content.append({
                    "title": result.get("title", urlparse(url).netloc),
                    "source": result.get("source", result.get("source_name", urlparse(url).netloc)),
                    "content": snippet, # Use the snippet as content
                    "url": url,
                    "is_news": result.get("is_news", False), # Keep news flag
                    "extraction_method": "snippet" # Indicate snippet was used
                 })
            else:
                 logging.warning(f"No snippet found for result {i+1}/{len(search_results)}: {url}. Skipping.")
                 # We could add the result with empty content, but it's unlikely to be useful for analysis.

            # No need for delays between scraping, as we're not scraping pages.
            # Delay is handled by SerpApi call implicitly.


        logging.info(f"Finished snippet extraction. Extracted content from {len(extracted_content)} sources.")
        return extracted_content

    def analyze_content(self, extracted_content: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Analyze and filter content (snippets) for relevance using AI."""
        analyzed_content = []

        logging.info(f"Analyzing relevance and extracting key points for {len(extracted_content)} snippets.")

        # Filter out content that is too short to be meaningful snippets
        processable_content = [item for item in extracted_content if item["content"] and len(item["content"]) > 20] # Require minimum snippet length

        for i, content_item in enumerate(processable_content):
            logging.info(f"Analyzing snippet {i+1}/{len(processable_content)} from {content_item.get('url', 'N/A')}")

            relevance = self.content_analyzer.analyze_relevance(content_item["content"], query)
            key_points = self.content_analyzer.extract_key_points(content_item["content"], query)

            analyzed_content.append({
                **content_item,
                "relevance": relevance,
                "key_points": key_points
            })
            # Small delay between AI calls if needed, but Gemini is usually fast.
            # time.sleep(0.1)

        # Sort content by relevance, highest first
        analyzed_content.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)

        logging.info(f"Finished content analysis. {len(analyzed_content)} snippets analyzed.")
        return analyzed_content

    def synthesize_information(self, analyzed_content: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Use AI to synthesize information from snippets into a final report."""
        logging.info(f"Synthesizing report for query: '{query}'")
        try:
            # Select the most relevant snippets for synthesis
            # Consider a threshold instead of just top N, or a mix
            relevant_content = [item for item in analyzed_content if item.get("relevance", 0.0) > 0.3] # Threshold
            # Also take top N if few meet threshold, ensure at least one source if any exist
            top_n = 7 # Synthesize from up to 7 relevant snippets
            if len(relevant_content) < min(top_n, len(analyzed_content)):
                 relevant_content = analyzed_content[:top_n]

            if not relevant_content:
                 logging.warning("No relevant snippets found (relevance < 0.3 or less than 7 top results). Generating limited report.")
                 # Generate a minimal report indicating no relevant info was found from snippets.
                 return {
                    "query": query,
                    "summary": f"Research on '{query}' yielded no highly relevant information from search result snippets.",
                    "sources": analyzed_content, # Show analyzed sources even if low relevance
                    "content": f"# Research Report: {query}\n\n## Executive Summary\nResearch conducted for the query \"{query}\" did not find sufficient highly relevant information within the provided search result snippets to generate a detailed report.\n\n## Conclusion\nThe analysis of search result snippets did not identify significant content directly addressing the query. A full web scrape might be needed for a more comprehensive answer."
                 }


            # Prepare context for the AI model from snippets and key points
            context = []
            sources_list = []

            for i, item in enumerate(relevant_content):
                 # Include title, source, URL, and key points/snippet
                 content_desc = ""
                 if item.get("key_points") and "no specific key points" not in item["key_points"][0].lower():
                     content_desc = "Key points from snippet: " + "; ".join(item["key_points"])
                     # Truncate key points if combined length is huge
                     if len(content_desc) > 1000: content_desc = content_desc[:1000] + "..."
                 else:
                     # Use the raw snippet if key points extraction failed or was generic
                     content_desc = "Snippet: " + item["content"]
                     if len(content_desc) > 1000: content_desc = content_desc[:1000] + "..." # Truncate raw snippet

                 context.append(f"Source {i+1}: {item.get('title', 'Untitled')}\nURL: {item.get('url', 'N/A')}\nRelevance: {item.get('relevance', 'N/A'):.2f}\nContent Summary/Key Points (from snippet): {content_desc}")

                 sources_list.append({
                     "title": item.get("title", "Untitled Source"),
                     "url": item.get("url", "#"),
                     "relevance": item.get("relevance", 0.0),
                     "source_type": "News" if item.get("is_news", False) else "Web Snippet",
                     "extraction_method": item.get("extraction_method", "snippet")
                 })

            context_text = "\n\n---\n\n".join(context)

            prompt = f"""
            Task: Create a comprehensive, well-structured research report based *only* on the provided information from the search result snippets.
            Synthesize the key findings and information points to answer the Research Query.
            Acknowledge that the information comes from snippets, which are brief summaries.
            If information is conflicting or uncertain across snippets, mention that. Avoid making claims not supported by the snippets.
            Include citations by referring to Source numbers (e.g., according to Source 3...).

            Research Query: "{query}"

            Information from search result snippets:
            {context_text}

            Instructions:
            1.  Structure the report with the following Markdown sections:
                `# Research Report: [Query] (Based on Search Snippets)`
                `## Executive Summary` (1-3 sentences summarizing the main answer/findings based *only* on the snippets.)
                `## Key Findings from Snippets` (Bulleted or numbered list of the most important points relevant to the query, synthesized from multiple snippets where possible. Cite sources.)
                `## Analysis (Based on Snippets)` (Discuss the findings in more detail, compare information across snippets if relevant, explain complexities or limitations due to using only snippets.)
                `## Conclusion` (Summarize the report's main points and briefly state what the snippets indicate about the query. Mention this is based on snippets.)
                `## Sources (Snippets Used)` (List the sources used with their titles, URLs, and relevance scores.)
            2.  Use clear and concise language.
            3.  Do not include external information not present in the provided snippets.
            4.  Ensure factual accuracy based *only* on the provided snippet data.
            5.  Format the source list clearly in the "Sources" section.
            6.  Cite sources near the points they support in Key Findings or Analysis.

            Generate the full report in Markdown format.
            """

            logging.info("Sending synthesis prompt to AI model...")
            response = self.report_model.generate_content(prompt)
            report_content = response.text.strip()
            logging.info("AI synthesis complete.")

            # Post-process the report content to ensure sources are listed correctly
            sources_section_title = "## Sources (Snippets Used)"
            if sources_section_title not in report_content:
                 report_content += f"\n\n{sources_section_title}\n"
                 for i, source in enumerate(sources_list):
                     report_content += f"{i+1}. [{source['title']}]({source['url']}) - Relevance: {source['relevance']:.2f} ({source['source_type']})\n"
            else:
                # Find the sources section and replace/append if needed
                sources_start = report_content.find(sources_section_title)
                sources_end = report_content.find("\n#", sources_start + len(sources_section_title)) # Find next section title starting with #

                current_sources_text = report_content[sources_start:] if sources_end == -1 else report_content[sources_start:sources_end]

                # Simple check: If the section seems empty or malformed, regenerate it
                if len(current_sources_text.splitlines()) < 3 or "http" not in current_sources_text:
                     new_sources_text = f"{sources_section_title}\n"
                     for i, source in enumerate(sources_list):
                         new_sources_text += f"{i+1}. [{source['title']}]({source['url']}) - Relevance: {source['relevance']:.2f} ({source['source_type']})\n"

                     if sources_end == -1:
                          report_content = report_content[:sources_start] + new_sources_text
                     else:
                          report_content = report_content[:sources_start] + new_sources_text + report_content[sources_end:]


            # Attempt to extract summary for the return dictionary
            summary_match = re.search(r'## Executive Summary\s*\n+(.*?)(?:\n\n|\n##|$)', report_content, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else "Research report generated based on available search snippets."


            return {
                "query": query,
                "summary": summary,
                "sources": sources_list,
                "content": report_content
            }

        except Exception as e:
            logging.error(f"Error in synthesize_information AI call: {str(e)}. Falling back to basic report generation from snippets.")

            # Fallback Report Generation
            all_key_points = []
            sources_list = []

            # Use some of the most relevant snippets for the fallback
            fallback_sources = analyzed_content[:min(len(analyzed_content), 5)]

            if not fallback_sources:
                 return {
                    "query": query,
                    "summary": f"Research on '{query}' encountered an error during synthesis.",
                    "sources": analyzed_content,
                    "content": f"# Research Report: {query} (Synthesis Error)\n\n## Executive Summary\nA synthesis error occurred while processing the search result snippets. Unable to generate a detailed report."
                 }


            for item in fallback_sources:
                if item.get("key_points") and "no specific key points" not in item["key_points"][0].lower():
                    all_key_points.extend(item["key_points"])
                elif item.get("content"):
                     # If no key points, take the snippet directly
                     if item["content"].strip(): all_key_points.append(f"Snippet: {item['content']}")


                sources_list.append({
                    "title": item.get("title", "Untitled Source"),
                    "url": item.get("url", "#"),
                    "relevance": item.get("relevance", 0.0),
                    "source_type": "News" if item.get("is_news", False) else "Web Snippet",
                    "extraction_method": item.get("extraction_method", "snippet")
                })

            summary = f"Research on '{query}' encountered an error during full synthesis. Basic findings from {len(sources_list)} snippets are listed."

            content_sections = [
                f"# Research Report: {query} (Fallback from Snippets)",
                f"\n## Summary\n{summary}",
                "\n## Key Findings (Basic Fallback from Snippets)"
            ]

            if all_key_points:
                 for i, point in enumerate(all_key_points):
                     content_sections.append(f"- {point}") # Use bullet points for fallback
            else:
                 content_sections.append("No specific key points could be extracted by the fallback from snippets.")

            content_sections.append("\n## Sources (Snippets Used)")
            for i, source in enumerate(sources_list):
                content_sections.append(f"- [{source['title']}]({source['url']}) - Relevance: {source['relevance']:.2f} ({source['source_type']})")


            full_content = "\n".join(content_sections)

            return {
                "query": query,
                "summary": summary,
                "sources": sources_list,
                "content": full_content
            }


    def research(self, query: str) -> Dict[str, Any]:
        """Execute the full research pipeline."""
        try:
            logging.info(f"Starting research for query: '{query}'")

            # Analyze the query
            analyzed_query = self.analyze_query(query)
            print(f"\nAnalyzed query: {analyzed_query}")

            # Perform the web search
            search_results = self.search_web(analyzed_query)
            print(f"Found {len(search_results)} initial search results.")

            # Extract content (snippets) from search results
            extracted_content = self.extract_content(search_results)
            # Filter out results where snippet extraction failed
            available_snippets = [item for item in extracted_content if item["content"]]
            print(f"Found and extracted snippets from {len(available_snippets)} sources.")

            if not available_snippets:
                 logging.warning("No usable snippets found in search results.")
                 return {
                    "status": "success",
                    "report": {
                        "query": query,
                        "summary": f"Research on '{query}' could not find usable snippets in any search results.",
                        "sources": extracted_content, # Show results even if no snippet was found
                        "content": f"# Research Report: {query}\n\n## Executive Summary\nThe research agent was unable to find usable snippets in the search results for the query \"{query}\". This report is based on the attempt to find snippets.\n\n## Conclusion\nNo report could be generated due to the lack of available snippets in the search results."
                    }
                 }


            # Analyze content (snippets) for relevance and key points
            analyzed_content = self.analyze_content(available_snippets, query)
            print(f"Analyzed snippets for relevance and extracted key points.")

            # Synthesize information into a report
            report = self.synthesize_information(analyzed_content, query)
            print(f"Generated research report.")

            return {
                "status": "success",
                "report": report
            }

        except Exception as e:
            error_message = f"An unexpected error occurred during research: {str(e)}"
            logging.error(error_message, exc_info=True) # Log traceback
            return {
                "status": "error",
                "message": error_message
            }


# --- Main Execution ---

# def main():
#     """Main function to run the research agent."""
#     # No need for async or context manager without Playwright
#     agent = WebResearchAgent(SERPAPI_API_KEY)

#     query = input("Enter your research query: ")
#     if not query:
#         query = "What are the latest developments in renewable energy?"
#     print(f"\nResearching: '{query}'")

#     result = agent.research(query) # Call the synchronous research method

#     if result["status"] == "success":
#         print("\n" + "=" * 70) # Use longer separator
#         print(result["report"]["content"])
#         print("=" * 70)
#     else:
#         print(f"\nError: {result['message']}")

# if __name__ == "__main__":
#     # Run the synchronous main function
#     main()

# if __name__ == "__main__":
#     import argparse
 
#     parser = argparse.ArgumentParser(description="Run the research agent with a question.")
#     parser.add_argument('--question', type=str, help='Research question to answer')
 
#     args = parser.parse_args()
#     query = args.question or input("Enter your research query: ").strip()
#     agent = WebResearchAgent(SERPAPI_API_KEY)
#     if not query:
#         query = "What are the latest developments in renewable energy?"
 
#     print(f"\nResearching: '{query}'")
#     response = agent.research(query)
 
#     if response["status"] == "success":
#         print("\n" + "=" * 70)
#         print(response["report"]["content"])
#         print("=" * 70)
#     else:
#         print(f"\nError: {response['message']}")



# run4.py

def main(query):
    # replace with actual import
 # Replace or load from env/secure place

    if not query:
        query = "What are the latest developments in renewable energy?"

    agent = WebResearchAgent(SERPAPI_API_KEY)
    print(f"\nResearching: '{query}'")
    response = agent.research(query)

    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the research agent with a question.")
    parser.add_argument('--question', type=str, help='Research question to answer')

    args = parser.parse_args()
    query = args.question or input("Enter your research query: ").strip()

    result = main(query)

    if result["status"] == "success":
        print("\n" + "=" * 70)
        print(result["report"]["content"])
        print("=" * 70)
    else:
        print(f"\nError: {result['message']}")
