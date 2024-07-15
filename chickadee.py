import json
import asyncio
from typing import List, Dict, Any, Optional
from aiofile import AIOFile
from openai import AsyncOpenAI
import logfire
import os
import tiktoken
from pydantic import BaseModel, Field
import instructor

# Initialize logfire
logfire.configure()

# Initialize AsyncOpenAI client with Instructor
client = instructor.patch(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Initialize tiktoken encoder
encoding = tiktoken.encoding_for_model("gpt-4o")


class Prompt(BaseModel):
    role: str = Field(..., description="Define the AI's role")
    instructions: str = Field(..., description="Provide clear instructions")
    steps: str = Field(..., description="Outline specific steps")
    end_goal: str = Field(..., description="State the goal and audience")
    narrowing: str = Field(..., description="Add constraints")
    reasoning: str = Field(
        ...,
        description="Explanation of the thought process behind creating this prompt",
    )


class AnalysisResult(BaseModel):
    analysis: str = Field(
        ...,
        description="Detailed breakdown of common themes, patterns, and intents identified in the questions",
    )
    prompts: List[Prompt] = Field(
        ..., description="List of reusable LLM prompts based on the analysis"
    )


def num_tokens_from_string(string: str) -> int:
    return len(encoding.encode(string))


async def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    async with AIOFile(file_path, "r") as file:
        content = await file.read()
        return json.loads(content)


async def extract_first_question(conversation: Dict[str, Any]) -> Optional[str]:
    mapping = conversation.get("mapping", {})
    for message_data in mapping.values():
        if not isinstance(message_data, dict):
            continue
        message = message_data.get("message")
        if not isinstance(message, dict):
            continue
        author = message.get("author", {})
        if not isinstance(author, dict):
            continue
        if author.get("role") == "user":
            content = message.get("content", {})
            if not isinstance(content, dict):
                continue
            if content.get("content_type") == "text":
                parts = content.get("parts", [])
                if parts and isinstance(parts[0], str):
                    question = parts[0]
                    if question and not question.strip().lower().startswith(
                        "act as a search copilot"
                    ):
                        return question
    return None


async def analyze_questions_chunk(questions: List[str]) -> AnalysisResult:
    prompt = f"""
    Role: You are an Expert Prompt Engineer specializing in analyzing user queries and creating reusable, high-quality prompts for language models.

    Task: Analyze the following list of questions from ChatGPT conversations and create reusable prompts based on your analysis using the RISEN framework.

    Questions to analyze:
    {questions}

    Instructions:
    1. Identify common themes and patterns across the questions.
    2. Create 3-5 general, widely applicable prompts that address these themes.
    3. For each prompt, provide a brief explanation of your thought process.
    4. Structure each prompt using the RISEN framework:
       - R — Role: Define the AI's role.
       - I — Instructions: Provide clear instructions.
       - S — Steps: Outline specific steps.
       - E — End goal: State the goal and audience.
       - N — Narrowing: Add constraints.

    Output:
    Provide your analysis and generated prompts in a structured format with 'analysis' and 'prompts' sections. Each prompt should have 'role', 'instructions', 'steps', 'end_goal', 'narrowing', and 'reasoning' fields.
    """

    logfire.info(f"Sending request to OpenAI API {len(questions)}")
    response = await client.chat.completions.create(
        model="gpt-4o",
        response_model=AnalysisResult,
        messages=[
            {
                "role": "system",
                "content": "You are an Expert Prompt Engineer with extensive experience in analyzing user queries and creating high-quality, reusable prompts for language models using the RISEN framework.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    logfire.info("Received response from OpenAI API")
    logfire.info(f"Analysis result {len(response.prompts)}")
    return response


async def aggregate_results(all_results: List[AnalysisResult]) -> AnalysisResult:
    combined_prompts = []
    for result in all_results:
        combined_prompts.extend(result.prompts)

    if not combined_prompts:
        logfire.error("No prompts found to aggregate.")
        return AnalysisResult(
            analysis="No prompts were available for aggregation.", prompts=[]
        )

    prompt = f"""
    Role: You are an AI Aggregator and Refiner, specializing in consolidating and improving AI-generated content.

    Task: Review and refine the following set of prompts generated from multiple analyses of user questions. Your goal is to eliminate redundancy, combine similar prompts, and create a concise, high-quality set of reusable prompts using the RISEN framework.

    Prompts to refine:
    {json.dumps([prompt.dict() for prompt in combined_prompts], indent=2)}

    Instructions:
    1. Identify common themes and patterns across the prompts.
    2. Combine similar prompts into more general, widely applicable versions.
    3. Eliminate redundancy while preserving unique insights.
    4. Ensure each refined prompt is clear, concise, and effective.
    5. Structure each refined prompt using the RISEN framework:
       - R — Role: Define the AI's role.
       - I — Instructions: Provide clear instructions.
       - S — Steps: Outline specific steps.
       - E — End goal: State the goal and audience.
       - N — Narrowing: Add constraints.
    6. Provide a brief explanation for each refined prompt, highlighting its purpose and applicability.

    Output:
    You MUST provide your output in the following JSON structure:
    {{
        "analysis": "Your overall analysis of the prompts and refinement process",
        "prompts": [
            {{
                "role": "The AI's role",
                "instructions": "Clear instructions",
                "steps": "Specific steps",
                "end_goal": "Goal and audience",
                "narrowing": "Constraints",
                "reasoning": "Your explanation for this refined prompt"
            }},
            // ... more prompts ...
        ]
    }}

    Ensure that your response can be parsed as valid JSON with 'analysis' and 'prompts' fields, where 'prompts' is a list of objects each containing 'role', 'instructions', 'steps', 'end_goal', 'narrowing', and 'reasoning' fields.
    """

    logfire.info("Sending aggregation request to OpenAI API")
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            response_model=AnalysisResult,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI Aggregator and Refiner with expertise in consolidating and improving AI-generated content using the RISEN framework. Always provide your response in the requested JSON structure.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        logfire.info("Received aggregation response from OpenAI API")
        logfire.info(f"Aggregation result: {len(response.prompts)} refined prompts")
        return response
    except Exception as e:
        logfire.error(f"Error in aggregation: {str(e)}")
        return AnalysisResult(
            analysis="Error in aggregating results. Please check the logs for more information.",
            prompts=[],
        )


async def main():
    file_path = "conversations.json"
    all_results = []
    all_prompts = []
    max_tokens = (
        6000  # Leave some room for the system message and other parts of the prompt
    )
    current_batch = []
    current_batch_tokens = 0

    logfire.info(f"Starting processing {file_path}", file_path=file_path)

    conversations = await load_conversations(file_path)

    len_conversations = len(conversations)
    logfire.info(
        f"Total conversations to process: {len_conversations}",
    )

    for i, conversation in enumerate(conversations, 1):
        total = (len(conversations),)
        conversation_id = conversation.get("id", "unknown")
        logfire.info(f"Processing conversation {i} of {total} - ID: {conversation_id}")
        try:
            question = await extract_first_question(conversation)
            if question:
                logfire.debug(f"Extracted question: {question}", question=question)
                question_tokens = num_tokens_from_string(question)
                if current_batch_tokens + question_tokens > max_tokens:
                    # Process the current batch before adding this question
                    result = await analyze_questions_chunk(current_batch)
                    all_results.append(result)
                    all_prompts.extend(result.prompts)
                    question_count = len(current_batch)
                    prompt_count = len(result.prompts)
                    logfire.info(
                        f"Batch processed {question_count} questions, {prompt_count} prompts"
                    )
                    current_batch = []
                    current_batch_tokens = 0

                current_batch.append(question)
                current_batch_tokens += question_tokens
            else:
                logfire.warn(
                    f"No valid user question found in conversation {conversation_id}",
                    conversation_id=conversation.get("id", "unknown"),
                )
        except Exception as e:
            logfire.error(f"Error processing conversation {i}: {e}")

    # Process any remaining questions in the last batch
    if current_batch:
        result = await analyze_questions_chunk(current_batch)
        all_results.append(result)
        all_prompts.extend(result.prompts)
        question_count = len(current_batch)
        prompt_count = len(result.prompts)
        logfire.info(
            f"Final batch processed {question_count} questions, {prompt_count} prompts"
        )

    # Aggregate and refine results
    refined_results = await aggregate_results(all_results)

    # Write refined prompts to file
    output_file = "refined_prompts.txt"
    write_prompts_to_file(refined_results.prompts, output_file)

    # Log summary
    total_conversations = (len(conversations),)
    total_initial_prompts = sum(len(result.prompts) for result in all_results)
    total_refined_prompts = len(refined_results.prompts)
    logfire.info(
        f"Processing summary - Conversations: {total_conversations}, Initial Prompts: {total_initial_prompts}, Refined Prompts: {total_refined_prompts}"
    )
    logfire.info(f"Refined prompts written to file {file_path}", file_path=output_file)

    # Write the analysis and prompts to separate files
    analysis_file = "refinement_analysis.txt"
    prompts_file = "refined_prompts.txt"

    with open(analysis_file, "w") as f:
        f.write(refined_results.analysis)
    logfire.info(f"Refinement analysis written to file {analysis_file}")

    write_prompts_to_file(refined_results.prompts, prompts_file)
    logfire.info(f"Refined prompts written to file {prompts_file}")


def write_prompts_to_file(prompts: List[Prompt], file_path: str):
    with open(file_path, "w") as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"Prompt {i}:\n")
            f.write(f"Role: {prompt.role}\n")
            f.write(f"Instructions: {prompt.instructions}\n")
            f.write(f"Steps: {prompt.steps}\n")
            f.write(f"End Goal: {prompt.end_goal}\n")
            f.write(f"Narrowing: {prompt.narrowing}\n")
            f.write(f"Reasoning: {prompt.reasoning}\n\n")
    logfire.info(f"Generated prompts written to {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
