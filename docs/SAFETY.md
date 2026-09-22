# Safety Model

NeurOS starts as a non-invasive software research prototype.

## Hard rules
1. No brain stimulation.
2. No surgical or implant instructions.
3. Raw EEG is sensitive biometric data and should be local/private by default.
4. Decoder output is probabilistic classification, not a statement of a user's thought.
5. Low-confidence classifications are discarded.
6. High-impact actions require confirmation from a non-neural channel.
7. The AI layer receives structured intent events rather than raw EEG by default.
8. Automated actions must be auditable and reversible where practical.

High-impact examples include payments, messages/publication, destructive file/account actions,
security changes, legal commitments, and medical actions. EEG alone never authorizes them.
