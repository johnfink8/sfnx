# Security

sFNX is made for a trusted network: one operator, their own GGUFs, configs and prompts, and clients they trust. Know
what that means before you expose it.

- **The installed service listens on every interface.** The unit `make install` writes runs the front end with
  `--host 0.0.0.0 --port 8080`, and `--host` overrides the config's `host`. Run by hand without `--host`, the front end
  takes the config's `host`, else 127.0.0.1. Edit the unit (`serve/sfnx.service.in`) to listen on one address.
- **Authentication is optional.** With `api_key` (or `STRATA_API_KEY`) every request needs a key. Without one, anyone
  who can reach the port can use the model. The only checks then are against web pages, not against clients:
  - **Host** (DNS rebinding): a request passes when its Host header is an IP address, `localhost` or a `*.localhost`
    name, a name in `allowed_hosts` (or `STRATA_ALLOWED_HOSTS`; `"*"` passes every name), or missing; and, listening
    beyond loopback, the machine's name, its LAN addresses and `host.docker.internal`.
  - **Origin**: a `/v1` POST from a browser page needs a JSON body and an origin that is this server's own page or a
    page on one of those names (an IP only when it is the request's own Host); a non-web scheme (a browser extension,
    an Electron app) passes. Clients that send no Origin (curl, the SDKs, other servers) pass.
- **Inputs are trusted.** GGUFs, configs, chat templates and pictures are not validated as hostile. Do not load files
  you did not make or check.
- **No sandbox.** The engine and the picture encoder run as the service's user, with the cards and memory it gives them.

To let anyone off your network reach it, set an `api_key`, or put it behind a reverse proxy with its own
authentication and TLS.

## Reporting a problem

A failure of one of the checks above (a key not checked, a Host or Origin passing that the rules above refuse) or a
crash from a request a trusted client could send: report it privately through the repository's **Security** tab,
**Report a vulnerability**. No authentication without a key, and no validation of the operator's own files, is the
design, not a vulnerability.
