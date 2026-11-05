def test_public_exports():
    from ntt_downstream.common.results import (
        SCHEMA_VERSION,
        MetricBlock,
        MetricEntry,
        MetricSnapshotCallback,
        ResultsDocument,
        RunMetadata,
        results_filename,
        write_results,
    )

    assert SCHEMA_VERSION == 2
    assert callable(write_results)
    assert results_filename("decoding") == "downstream_decoding_results.json"
    assert MetricEntry and MetricBlock and RunMetadata and ResultsDocument
    assert MetricSnapshotCallback
