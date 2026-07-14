"""Couchbase connection and one-time provisioning (Chapter 2)."""

import logging
import threading
from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (AuthenticationException,
                                  BucketNotFoundException,
                                  CollectionAlreadyExistsException,
                                  CouchbaseException,
                                  ScopeAlreadyExistsException)
from couchbase.options import ClusterOptions, KnownConfigProfiles

from . import config

logger = logging.getLogger(__name__)

_cluster: Cluster | None = None
_cluster_lock = threading.Lock()


def cluster() -> Cluster:
    global _cluster
    if _cluster is None:
        with _cluster_lock:  # concurrent first requests must not open duplicate connections
            if _cluster is None:
                opts = ClusterOptions(
                    PasswordAuthenticator(config.CB_USERNAME, config.CB_PASSWORD))
                if config.CB_CONN_STRING.startswith("couchbases://"):
                    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
                try:
                    _cluster = Cluster.connect(config.CB_CONN_STRING, opts)
                    _cluster.wait_until_ready(timedelta(seconds=10))
                except AuthenticationException as e:
                    raise RuntimeError(
                        f"Couchbase rejected CB_USERNAME={config.CB_USERNAME!r} for "
                        f"{config.CB_CONN_STRING!r} — check CB_USERNAME/CB_PASSWORD "
                        "in your .env. See docs/troubleshooting.md."
                    ) from e
                except CouchbaseException as e:
                    raise RuntimeError(
                        f"Couldn't connect to Couchbase at "
                        f"{config.CB_CONN_STRING!r}: {e}. Check the cluster is "
                        "running/reachable and CB_CONN_STRING is correct (use "
                        "couchbases:// for Capella, couchbase:// for local). "
                        "See docs/troubleshooting.md."
                    ) from e
    return _cluster


def bucket():
    return cluster().bucket(config.CB_BUCKET)


def ensure_collections() -> None:
    """Idempotent provisioning of every keyspace the service touches."""
    cm = bucket().collections()
    layout = [
        (config.DOCS_SCOPE, config.CHUNKS_COLLECTION, None),
        (config.DOCS_SCOPE, "semantic_cache", None),
        (config.CHAT_SCOPE, config.CHAT_COLLECTION, timedelta(days=30)),
        (config.EVALS_SCOPE, config.SAMPLES_COLLECTION, None),
    ]
    from couchbase.management.collections import CreateCollectionSettings

    for scope_name, coll_name, ttl in layout:
        try:
            cm.create_scope(scope_name)
        except ScopeAlreadyExistsException:
            pass
        except (BucketNotFoundException, CouchbaseException) as e:
            raise RuntimeError(
                f"Couldn't provision scope {scope_name!r} in bucket "
                f"{config.CB_BUCKET!r}: {e}. Most likely the {config.CB_BUCKET!r} "
                "bucket doesn't exist yet — create it in the Couchbase Server/Capella "
                "UI first (this app only provisions scopes/collections inside an "
                "existing bucket, same as notebook 01). See docs/troubleshooting.md."
            ) from e
        try:
            settings = CreateCollectionSettings(max_expiry=ttl) if ttl else None
            cm.create_collection(scope_name, coll_name, settings)
        except CollectionAlreadyExistsException:
            pass
    logger.info("Provisioned/verified collections in bucket %r", config.CB_BUCKET)
