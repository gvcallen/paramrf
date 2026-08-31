# Changelog

## Unreleased

- Correct microstrip results by enabling Kirschning--Jansen modal dispersion by
  default. Set `dispersion=None` on `MicrostripLine` to retain the quasi-static
  pipeline explicitly.
- Add the Hammerstad--Jensen microstrip formulation and selectable complex
  (ADS-like) and real (QUCS-like) permittivity conventions.
