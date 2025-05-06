'''
title: YandexGPT RAG Pipeline
author: butterboard
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information using YandexGPT and Yandex Cloud ML Search Index.
requirements:
  - yandex-cloud-ml-sdk
  - python-dotenv
'''

from typing import List, Union, Generator, Iterator
import os
import pathlib
from schemas import OpenAIChatMessage
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    StaticIndexChunkingStrategy,
    TextSearchIndexType,
)


class Pipeline:
    def __init__(self):
        self.sdk = None
        self.assistant = None
        self.search_index = None
        self.files = []

    async def on_startup(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        folder_id = os.getenv("YANDEX_FOLDER_ID")
        api_key = os.getenv("YANDEX_API_KEY")

        if not folder_id or not api_key:
            raise ValueError("Missing Yandex Cloud credentials in environment variables")

        # Initialize Yandex SDK
        self.sdk = YCloudML(
            folder_id=folder_id,
            auth=api_key,
        )

        # Load and upload documents
        data_path = pathlib.Path("./data")
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found at {data_path}")

        for file_path in data_path.iterdir():
            if file_path.is_file():
                file = self.sdk.files.upload(
                    file_path,
                    ttl_days=5,
                    expiration_policy="static",
                )
                self.files.append(file)

        # Create search index
        operation = self.sdk.search_indexes.create_deferred(
            self.files,
            index_type=TextSearchIndexType(
                chunking_strategy=StaticIndexChunkingStrategy(
                    max_chunk_size_tokens=700,
                    chunk_overlap_tokens=300,
                )
            ),
        )
        self.search_index = operation.wait()

        # Create assistant with search tool
        search_tool = self.sdk.tools.search_index(self.search_index)
        self.assistant = self.sdk.assistants.create(
            "yandexgpt",
            tools=[search_tool],
        )

    async def on_shutdown(self):
        # Cleanup resources
        if self.assistant:
            self.assistant.delete()
        if self.search_index:
            self.search_index.delete()
        for file in self.files:
            file.delete()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Create new thread for each request
        thread = self.sdk.threads.create()

        try:
            # Add conversation history
            for msg in messages:
                if msg.get("role") == "user":
                    thread.write(msg.get("content"))

            # Add current message
            thread.write(user_message)

            # Execute assistant
            run = self.assistant.run(thread)
            result = run.wait()

            # Return response as generator
            def response_generator():
                yield result.text

            return response_generator()
        finally:
            # Cleanup thread
            thread.delete()