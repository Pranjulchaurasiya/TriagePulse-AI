"""Comprehensive Clinical Red-Flag Evaluation Suite.
Tests 100+ variations covering all emergency categories, verifying a strict 0% False Negative Rate.
"""

from __future__ import annotations
import pytest
from safety.red_flag_gate import RedFlagGate

gate = RedFlagGate()

# 60+ Positive Emergency Phrases (Target: 100% detection, 0% False Negatives)
POSITIVE_EMERGENCY_CASES = [
    # Cardiac
    "I have severe chest pain and dizziness",
    "There is crushing pain in my chest right now",
    "I feel a tight pressure in chest",
    "It feels like an elephant on my chest",
    "I think I am having a heart attack",
    "There is a severe chest tightness since 10 minutes",
    "I have heavy chest sensations",
    "The pain is radiating to my left arm",
    "Pain shooting to left arm and jaw",
    "Severe heart palpitations and almost fainting",
    "My husband has pain in his chest",
    "She has tightness in her chest",
    # Respiratory
    "I can't breathe please help me",
    "I cannot breathe properly at all",
    "cant breathe need emergency",
    "My mother is struggling to breathe right now",
    "He is fighting for breath on the sofa",
    "I am gasping for air",
    "He is choking on a piece of food",
    "The baby has blue lips and is limp",
    "His lips turning blue quickly",
    "Having a severe asthma attack and inhaler is not working",
    "My inhaler isn't working and I am wheezing heavily",
    "I feel my throat closing up after eating peanuts",
    "Help me I am suffocating",
    "My mother has severe difficulty breathing",
    "I am extremely short of breath right now",
    "He can't catch his breath at all",
    # Stroke / Neurological (FAST)
    "Her face is drooping on the left side",
    "I noticed a sudden facial droop in my dad",
    "His face drooped while talking",
    "He suddenly has slurred speech and can't speak clearly",
    "I am slurring my words and feel confused",
    "My grandmother can't speak at all",
    "He cannot speak properly since this morning",
    "I have sudden weakness in my right arm and leg",
    "Severe arm weakness on my left side",
    "I have numbness on one side of my body",
    "I can't move my arm suddenly",
    "She cannot move her leg at all",
    "Sudden paralysis on one side of face",
    "I think my husband is having a stroke",
    "The patient is having a violent seizure right now",
    "My child is having a fit on the floor",
    "He is fitting uncontrollably and shaking",
    "She is having epileptic fits",
    "My son is convulsing on the floor",
    # Severe Hemorrhage / Trauma
    "I am coughing up blood and feeling dizzy",
    "He has been vomiting blood repeatedly",
    "She is throwing up bright red blood",
    "Severe bleeding from a deep wound",
    "He is bleeding profusely from his leg",
    "Blood spurting from an open cut",
    "The bleeding won't stop after applying pressure for 20 minutes",
    "Bleeding will not stop and I feel faint",
    "My neighbor was stabbed in the street",
    "He has a gunshot wound to the leg",
    "There is an arterial bleed in his forearm",
    # Unresponsive / Collapse
    "My husband passed out on the kitchen floor",
    "She collapsed suddenly and is unresponsive",
    "The patient is unconscious and not responding to his name",
    "He is completely unresponsive to pain",
    "My grandfather is not waking up this morning",
    "She won't wake up no matter what I do",
    "I blacked out while driving",
    "He had a sudden loss of consciousness",
    # Anaphylaxis / Severe Allergy
    "I think I am in anaphylaxis please send help",
    "He is going into anaphylactic shock right now",
    "My tongue is swollen after eating walnuts",
    "Patient has a swollen tongue and cannot swallow",
    "Her throat is swollen and she can't breathe",
    "I am having a severe allergic reaction to penicillin",
    # Psychiatric Crisis / Overdose
    "I want to kill myself today",
    "I just want to end my life please",
    "I am planning to commit suicide",
    "I have taken an overdose of sleeping pills",
    "I swallowed a lot of pills an hour ago",
    "Someone cut my wrists in the bathroom",
    "My daughter wants to kill herself",
    "He wants to end his life today",
    "My brother took an overdose of paracetamol",
    "She cut her wrists and is bleeding",
    "My partner is feeling suicidal",
]

# 40+ Non-Emergency Controls (Negative Cases - False Positive Resistance)
NEGATIVE_CONTROL_CASES = [
    "I'd like to book a routine doctor appointment for next Tuesday",
    "Can I see Dr. Smith next week for a routine checkup?",
    "I have a sore throat and a runny nose for 3 days",
    "Need to check my blood test results from yesterday",
    "My arm hurts a bit after lifting heavy boxes yesterday",
    "Could I get a repeat prescription for my blood pressure tablets?",
    "I need to reschedule my hearing aid fitting for next Monday",
    "Can I book an appointment for a coil fitting with the nurse?",
    "I need to collect my fit note for work",
    "What are your surgery opening hours on Thursday?",
    "I need a sickness certificate for my employer",
    "My child has some mild chickenpox spots but is eating well",
    "Need to schedule an appointment with the practice nurse for a smear test",
    "Is it possible to register as a new NHS patient at your practice?",
    "Can I book a flu vaccination clinic for Saturday?",
    "I have a mild headache and tiredness since yesterday",
    "My back aches when I sit down at my desk",
    "Can the nurse change my surgical wound dressing tomorrow?",
    "I need to discuss my cholesterol medication with the doctor",
    "When is the phlebotomy blood clinic open in the morning?",
    "I have an itchy rash on my elbow that started last week",
    "Can I request a fit note extension please?",
    "I lost my eyeglasses and have a mild headache",
    "My knee feels stiff after running",
    "Need travel vaccinations before going to Spain next month",
    "Can I speak with a receptionist about changing my home address?",
    "My throat is a bit scratchy in the morning",
    "Need to book an annual diabetes review",
    "Can I have a telephone consultation about my test results?",
    "I would like an appointment to check an earwax blockage",
    "Can I get some advice on quitting smoking?",
    "My baby is due for 8 week immunisations",
    "Need advice on treating verrucas at home",
    "Can I pick up my prescription form at the front desk?",
    "I sprained my ankle 4 days ago and it is slowly improving",
    "I have mild eczema on my hands",
    "Can I get a doctor letter for the housing department?",
    "Is the surgery open over the Easter bank holiday?",
    "Can I book a routine blood pressure check with the healthcare assistant?",
    "I'd like to book a routine review for my asthma inhalers with the nurse",
    "My neck is slightly stiff from sleeping awkwardly",
    "How do I submit an online consultation form?",
    "Do you offer evening clinics for working patients?",
]


@pytest.mark.parametrize("emergency_phrase", POSITIVE_EMERGENCY_CASES)
def test_zero_false_negatives_on_emergencies(emergency_phrase: str):
    """CRITICAL SAFETY TEST: Verify 0% False Negatives on red-flag emergencies."""
    result = gate.check(emergency_phrase)
    assert result.triggered, (
        f"FALSE NEGATIVE DETECTED! Emergency query failed to trigger safety reflex: '{emergency_phrase}'"
    )
    assert result.category is not None
    assert result.latency_ms < 5.0, f"Safety gate took too long: {result.latency_ms}ms"


@pytest.mark.parametrize("normal_phrase", NEGATIVE_CONTROL_CASES)
def test_no_false_positives_on_routine_inquiries(normal_phrase: str):
    """Verify standard GP receptionist queries do not accidentally trigger the red-flag reflex."""
    result = gate.check(normal_phrase)
    assert not result.triggered, (
        f"FALSE POSITIVE DETECTED! Routine query mistakenly flagged as emergency: '{normal_phrase}' "
        f"[matched pattern: {result.matched_pattern}]"
    )


def test_safety_gate_latency_benchmark():
    """Verify that safety gate evaluation runs synchronously in sub-2ms."""
    import time
    iterations = 500
    start = time.perf_counter()
    for phrase in POSITIVE_EMERGENCY_CASES[:20]:
        for _ in range(iterations // 20):
            gate.check(phrase)
    total_time_ms = (time.perf_counter() - start) * 1000.0
    avg_latency_ms = total_time_ms / iterations
    assert avg_latency_ms < 2.0, f"Average safety gate latency exceeds 2ms: {avg_latency_ms:.3f}ms"
