"""Couchbase connection and one-time provisioning (Chapter 2)."""

import threading
from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (CollectionAlreadyExistsException,
                                  ScopeAlreadyExistsException)
from couchbase.options import ClusterOptions, KnownConfigProfiles

from . import config

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
                _cluster = Cluster.connect(config.CB_CONN_STRING, opts)
                _cluster.wait_until_ready(timedelta(seconds=10))
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
        try:
            settings = CreateCollectionSettings(max_expiry=ttl) if ttl else None
            cm.create_collection(scope_name, coll_name, settings)
        except CollectionAlreadyExistsException:
            pass
