# Corporate networks that re-sign HTTPS traffic break Python's bundled CA list
# (CERTIFICATE_VERIFY_FAILED when calling the LLM or WhatsApp). truststore makes Python
# use the operating system's certificate store instead, like browsers and curl do.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
