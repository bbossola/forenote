import numpy as np
import pytest

from forenote.embed.base import EmbedError
from forenote.embed.litert import LiteRtEmbedder
from forenote.models.manifest import load_manifest, model_for_tier

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def embedder(model_dir, tmp_path_factory):
    entry = model_for_tier(load_manifest(), "default", "embedding")
    path = model_dir(entry.id) / entry.files[0].name
    e = LiteRtEmbedder(path, entry.id, tmp_path_factory.mktemp("litert"), 4)
    e.load()
    yield e
    e.close()


def test_vectors_have_768_values_and_unit_length(embedder):
    v = embedder.embed_text(["The project budget grows in the third quarter."], "document")
    assert v.shape == (1, 768)
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-3


def test_the_query_finds_the_related_document(embedder):
    docs = embedder.embed_text(
        ["The budget for next year grows by ten percent.", "The cat sleeps on the warm mat all day."], "document"
    )
    q = embedder.embed_text(["How much money do we plan to spend?"], "query")[0]
    assert int(np.argmax(docs @ q)) == 0


def test_the_same_text_gives_the_same_vector(embedder):
    a = embedder.embed_text(["same text"], "document")
    b = embedder.embed_text(["same text"], "document")
    assert np.allclose(a, b, atol=1e-6)


def test_an_input_over_the_limit_raises_embed_error(embedder):
    with pytest.raises(EmbedError):
        embedder.embed_text(["word " * 3000], "document")
