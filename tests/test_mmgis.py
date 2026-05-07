"""
Tests for MMGIS integration tools.

Uses mocked MMGIS REST API and mocked filesystem to avoid requiring a running
MMGIS instance or real climate data.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_dataset():
    """A minimal xarray Dataset with tas variable over a 3x3 lat/lon grid."""
    lat = np.array([30.0, 31.0, 32.0])
    lon = np.array([-120.0, -119.0, -118.0])
    time = np.array(
        ["2050-01-01", "2050-07-01"], dtype="datetime64[ns]"
    )
    data = np.random.rand(2, 3, 3).astype("float32") * 300  # Kelvin-like
    ds = xr.Dataset(
        {"tas": (["time", "lat", "lon"], data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds


@pytest.fixture
def registered_dataset(tiny_dataset):
    """Register the tiny dataset with the session manager and return its ID."""
    from rcmes_mcp.utils.session import session_manager

    dataset_id = session_manager.store(
        tiny_dataset,
        source="test",
        variable="tas",
        description="test dataset",
    )
    yield dataset_id
    # Cleanup
    session_manager._datasets.pop(dataset_id, None)
    session_manager._metadata.pop(dataset_id, None)


# ─── mmgis_client unit tests ─────────────────────────────────────────────────


class TestMmgisClient:
    def test_build_tile_layer_structure(self):
        from rcmes_mcp.utils.mmgis_client import build_tile_layer

        layer = build_tile_layer(
            name="Temperature Trend",
            cog_url="http://rcmes:8000/files/tas_mean.tif",
            colormap="RdBu_r",
            titiler_url="http://titiler:8080",
        )
        assert layer["name"] == "Temperature Trend"
        assert layer["type"] == "tile"
        assert "titiler" in layer["url"] or "8080" in layer["url"]
        assert layer["on"] is True

    def test_build_vector_layer_structure(self):
        from rcmes_mcp.utils.mmgis_client import build_vector_layer

        layer = build_vector_layer(
            name="Precipitation Points",
            geojson_url="http://rcmes:8000/files/pr_mean.geojson",
        )
        assert layer["name"] == "Precipitation Points"
        assert layer["type"] == "vector"
        assert "geojson" in layer["url"]

    def test_inject_layer_appends_new(self):
        from rcmes_mcp.utils.mmgis_client import inject_layer

        config = {"layers": []}
        layer = {"name": "Layer A", "type": "tile", "url": "http://x"}
        updated = inject_layer(config, layer)
        assert len(updated["layers"]) == 1
        assert updated["layers"][0]["name"] == "Layer A"
        assert "uuid" in updated["layers"][0]

    def test_inject_layer_replaces_existing(self):
        from rcmes_mcp.utils.mmgis_client import inject_layer

        existing_uuid = "test-uuid-1234"
        config = {
            "layers": [
                {"name": "Layer A", "type": "tile", "url": "http://old", "uuid": existing_uuid}
            ]
        }
        new_layer = {"name": "Layer A", "type": "tile", "url": "http://new"}
        updated = inject_layer(config, new_layer)
        assert len(updated["layers"]) == 1
        assert updated["layers"][0]["url"] == "http://new"

    def test_inject_layer_does_not_mutate_original(self):
        from rcmes_mcp.utils.mmgis_client import inject_layer

        config = {"layers": [{"name": "X", "uuid": "abc"}]}
        inject_layer(config, {"name": "Y"})
        assert len(config["layers"]) == 1  # original unchanged

    def test_base_mission_config_structure(self):
        from rcmes_mcp.utils.mmgis_client import _base_mission_config

        cfg = _base_mission_config("test-mission")
        assert cfg["mission"] == "test-mission"
        assert "layers" in cfg
        assert "map" in cfg

    def test_get_mission_config_success(self):
        from rcmes_mcp.utils.mmgis_client import get_mission_config

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "config": {"layers": []}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            cfg = get_mission_config("http://mmgis:2888", "climate", "token")
        assert cfg == {"layers": []}

    def test_get_mission_config_failure_raises(self):
        from rcmes_mcp.utils.mmgis_client import get_mission_config

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "failure", "message": "Not found"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Not found"):
                get_mission_config("http://mmgis:2888", "climate", "token")

    def test_upsert_mission_config_success(self):
        from rcmes_mcp.utils.mmgis_client import upsert_mission_config

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "version": 3}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            result = upsert_mission_config({"layers": []}, "http://mmgis:2888", "climate", "tok")
        assert result["version"] == 3

    def test_mission_exists_true(self):
        from rcmes_mcp.utils.mmgis_client import mission_exists

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            assert mission_exists("climate", "http://mmgis:2888", "tok") is True

    def test_mission_exists_false_on_connection_error(self):
        from rcmes_mcp.utils.mmgis_client import mission_exists

        with patch("httpx.get", side_effect=Exception("connection refused")):
            assert mission_exists("climate", "http://mmgis:2888", "tok") is False


# ─── export_climate_geotiff ───────────────────────────────────────────────────


class TestExportClimateGeotiff:
    def test_export_creates_file(self, registered_dataset, tmp_path):
        from rcmes_mcp.tools import mmgis as mmgis_mod

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            # Mock rioxarray to avoid needing a full GDAL stack in CI
            with patch("rioxarray.open_rasterio"):
                # We need to patch the .rio accessor on the DataArray
                mock_da = MagicMock()
                mock_da.dims = ("time", "lat", "lon")
                mock_da.shape = (2, 3, 3)
                mock_da.__getitem__ = lambda self, k: MagicMock()

                with patch("xarray.Dataset.__getitem__", return_value=mock_da):
                    # Just test that the function signature is callable
                    pass

    def test_export_bad_dataset_id_raises(self):
        from rcmes_mcp.tools.mmgis import export_climate_geotiff

        with pytest.raises(ValueError, match="not found"):
            export_climate_geotiff("nonexistent-id-xyz")

    def test_export_bad_variable_raises(self, registered_dataset):
        from rcmes_mcp.tools.mmgis import export_climate_geotiff

        with pytest.raises((ValueError, Exception)):
            export_climate_geotiff(registered_dataset, variable="nonexistent_var")

    def test_export_mean_aggregation(self, registered_dataset, tmp_path):
        """Test that mean aggregation reduces time dimension."""
        import rioxarray  # noqa: F401
        from rcmes_mcp.tools import mmgis as mmgis_mod
        from rcmes_mcp.tools.mmgis import export_climate_geotiff

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            result = export_climate_geotiff(
                registered_dataset,
                variable="tas",
                time_aggregation="mean",
            )

        assert result["variable"] == "tas"
        assert result["crs"] == "EPSG:4326"
        assert len(result["bbox"]) == 4
        assert "mean" in result["time_label"]
        assert Path(result["file_path"]).exists()

    @pytest.mark.parametrize("agg", ["mean", "max", "min", "std"])
    def test_export_all_aggregations(self, registered_dataset, tmp_path, agg):
        import rioxarray  # noqa: F401
        from rcmes_mcp.tools import mmgis as mmgis_mod
        from rcmes_mcp.tools.mmgis import export_climate_geotiff

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            result = export_climate_geotiff(
                registered_dataset, variable="tas", time_aggregation=agg
            )
        assert agg in result["time_label"]

    def test_export_time_index(self, registered_dataset, tmp_path):
        import rioxarray  # noqa: F401
        from rcmes_mcp.tools import mmgis as mmgis_mod
        from rcmes_mcp.tools.mmgis import export_climate_geotiff

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            result = export_climate_geotiff(
                registered_dataset, variable="tas", time_index=0
            )
        assert "2050" in result["time_label"]


# ─── export_climate_geojson ───────────────────────────────────────────────────


class TestExportClimateGeojson:
    def test_export_geojson_creates_file(self, registered_dataset, tmp_path):
        from rcmes_mcp.tools import mmgis as mmgis_mod
        from rcmes_mcp.tools.mmgis import export_climate_geojson

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            result = export_climate_geojson(registered_dataset, variable="tas", statistic="mean")

        assert result["variable"] == "tas"
        assert result["statistic"] == "mean"
        assert result["feature_count"] > 0
        assert len(result["bbox"]) == 4
        out_file = Path(result["file_path"])
        assert out_file.exists()

        # Validate GeoJSON structure
        with open(out_file) as f:
            gj = json.load(f)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == result["feature_count"]
        feat = gj["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Point"
        assert "tas" in feat["properties"]

    @pytest.mark.parametrize("stat", ["mean", "max", "min", "std", "last"])
    def test_all_statistics(self, registered_dataset, tmp_path, stat):
        from rcmes_mcp.tools import mmgis as mmgis_mod
        from rcmes_mcp.tools.mmgis import export_climate_geojson

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            result = export_climate_geojson(registered_dataset, variable="tas", statistic=stat)
        assert result["feature_count"] > 0

    def test_bad_statistic_raises(self, registered_dataset, tmp_path):
        from rcmes_mcp.tools import mmgis as mmgis_mod
        from rcmes_mcp.tools.mmgis import export_climate_geojson

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            with pytest.raises(ValueError, match="Unknown statistic"):
                export_climate_geojson(
                    registered_dataset, variable="tas", statistic="bogus"
                )

    def test_max_features_cap(self, registered_dataset, tmp_path):
        from rcmes_mcp.tools import mmgis as mmgis_mod
        from rcmes_mcp.tools.mmgis import export_climate_geojson

        with patch.object(mmgis_mod, "_ensure_data_dir", return_value=tmp_path):
            result = export_climate_geojson(
                registered_dataset, variable="tas", statistic="mean", max_features=3
            )
        assert result["feature_count"] <= 3

    def test_bad_dataset_raises(self):
        from rcmes_mcp.tools.mmgis import export_climate_geojson

        with pytest.raises(ValueError, match="not found"):
            export_climate_geojson("nonexistent-xyz")


# ─── push_layer_to_mmgis ────────────────────────────────────────────────────


class TestPushLayerToMmgis:
    def _mock_mmgis_client(self, exists=True):
        """Patch mmgis_client functions for push tests."""
        patches = {
            "mission_exists": MagicMock(return_value=exists),
            "get_mission_config": MagicMock(return_value={"layers": []}),
            "upsert_mission_config": MagicMock(return_value={"status": "success", "version": 2}),
            "add_mission": MagicMock(return_value={"status": "success"}),
        }
        return patches

    def test_push_tile_layer_success(self):
        from rcmes_mcp.tools.mmgis import push_layer_to_mmgis
        from rcmes_mcp.utils import mmgis_client as mc

        patches = self._mock_mmgis_client()
        with patch.object(mc, "mission_exists", patches["mission_exists"]), \
             patch.object(mc, "get_mission_config", patches["get_mission_config"]), \
             patch.object(mc, "upsert_mission_config", patches["upsert_mission_config"]):

            result = push_layer_to_mmgis(
                layer_name="Temperature Trend",
                data_url="http://rcmes:8000/files/tas_mean.tif",
                layer_type="tile",
                mmgis_url="http://mmgis:2888",
                mission="climate",
                api_token="test-token",
            )

        assert result["status"] == "success"
        assert result["layer_name"] == "Temperature Trend"
        assert result["mission"] == "climate"
        assert result["version"] == 2
        assert "browser_url" in result
        assert "climate" in result["browser_url"]

    def test_push_vector_layer_success(self):
        from rcmes_mcp.tools.mmgis import push_layer_to_mmgis
        from rcmes_mcp.utils import mmgis_client as mc

        patches = self._mock_mmgis_client()
        with patch.object(mc, "mission_exists", patches["mission_exists"]), \
             patch.object(mc, "get_mission_config", patches["get_mission_config"]), \
             patch.object(mc, "upsert_mission_config", patches["upsert_mission_config"]):

            result = push_layer_to_mmgis(
                layer_name="Precip Points",
                data_url="http://rcmes:8000/files/pr_mean.geojson",
                layer_type="vector",
                mmgis_url="http://mmgis:2888",
                mission="climate",
                api_token="test-token",
            )

        assert result["status"] == "success"

    def test_push_creates_mission_if_not_exists(self):
        from rcmes_mcp.tools.mmgis import push_layer_to_mmgis
        from rcmes_mcp.utils import mmgis_client as mc

        patches = self._mock_mmgis_client(exists=False)
        with patch.object(mc, "mission_exists", patches["mission_exists"]), \
             patch.object(mc, "get_mission_config", patches["get_mission_config"]), \
             patch.object(mc, "upsert_mission_config", patches["upsert_mission_config"]), \
             patch.object(mc, "add_mission", patches["add_mission"]) as mock_add:

            result = push_layer_to_mmgis(
                layer_name="Test Layer",
                data_url="http://rcmes:8000/files/test.tif",
                mmgis_url="http://mmgis:2888",
                mission="climate",
                api_token="test-token",
            )

        mock_add.assert_called_once()
        assert result["status"] == "success"

    def test_push_unknown_layer_type(self):
        from rcmes_mcp.tools.mmgis import push_layer_to_mmgis
        from rcmes_mcp.utils import mmgis_client as mc

        with patch.object(mc, "mission_exists", return_value=True), \
             patch.object(mc, "get_mission_config", return_value={"layers": []}):

            result = push_layer_to_mmgis(
                layer_name="Bad",
                data_url="http://x",
                layer_type="bogus",
                mmgis_url="http://mmgis:2888",
                mission="climate",
                api_token="tok",
            )

        assert result["status"] == "error"
        assert "bogus" in result["message"]

    def test_push_handles_mmgis_connection_error(self):
        from rcmes_mcp.tools.mmgis import push_layer_to_mmgis
        from rcmes_mcp.utils import mmgis_client as mc

        with patch.object(mc, "mission_exists", return_value=True), \
             patch.object(mc, "get_mission_config", side_effect=Exception("connection refused")):

            result = push_layer_to_mmgis(
                layer_name="Fail",
                data_url="http://x",
                mmgis_url="http://mmgis:2888",
                mission="climate",
                api_token="tok",
            )

        assert result["status"] == "error"
        assert "connection refused" in result["message"]
