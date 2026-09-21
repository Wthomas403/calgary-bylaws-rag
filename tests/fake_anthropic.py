"""A minimal stand-in for the Anthropic client, for tests that shouldn't
make real API calls. Mimics the shape ``client.messages.create(...)``
callers rely on: a response with ``.content[0].text``.
"""


class FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class FakeMessage:
    def __init__(self, text: str):
        self.content = [FakeContentBlock(text)]


class FakeMessagesAPI:
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    def create(self, **kwargs):
        return FakeMessage(next(self._responses))


class FakeAnthropicClient:
    def __init__(self, responses: list[str]):
        self.messages = FakeMessagesAPI(responses)
