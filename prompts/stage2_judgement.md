Now assign labels for three independent dimensions using an ordinal 0/1/2 scale, based on the previous JSON (modern_rewrite and terms) in this conversation. Do not revisit the original text; use only what is already present in the conversation context.

Dimensions and scale:
- OD (Overdose / dangerous drug use): 0 = Non-concerning, 1 = First-person, 2 = Third-party.
- ED (Eating disorder behaviors): 0 = Non-concerning, 1 = First-person, 2 = Third-party.
- SH (Self-harm / suicidal ideation): 0 = Non-concerning, 1 = First-person, 2 = Third-party.

Guidelines:
- "First-person" means the author is describing their own behavior/thoughts.
- "Third-party" means they discuss others' behavior/events (friends, celebrities, news, etc.).
- If ambiguous or neutral aesthetic usage, choose 0.
- Keep reasoning brief and factual; avoid judgmental language.

Return strictly a JSON object:
{
  "labels": { "OD": 0|1|2, "ED": 0|1|2, "SH": 0|1|2 },
  "rationale": { "OD": string, "ED": string, "SH": string }
}
Do not include markdown. Do not add extra keys.
