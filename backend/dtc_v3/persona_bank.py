"""
backend/dtc_v3/persona_bank.py — Category-specific persona pools.

Per friend's Day 2 spec (2026-04-28):
  - Personas must be product-relevant, not generic.
  - 15 personas per sub-pool, 3-5 sub-pools per major category.
  - Determinism preserved: same (seed, agent_idx) → same persona.
  - Cross-product: different categories naturally use different banks.

Architecture:
  - Specialized banks (ENERGY_DRINK_*, HYDRATION_*, etc.) for known categories
  - Generic fallback banks (FOOD_BEVERAGE_GENERIC, etc.) for unmatched categories
  - GENERIC_PERSONAS as last-resort fallback

This file is read by persona_generator.py. Do not import from here directly
in discussion.py — go through the generator.
"""

from __future__ import annotations

# Each persona: (name, age, profession, segment, profile)
# Tuples for memory efficiency. Generator wraps to dicts.
Persona = tuple[str, int, str, str, str]


# ═══════════════════════════════════════════════════════════════════════
# ENERGY DRINK PERSONAS (4 sub-pools × 15 = 60 personas)
# ═══════════════════════════════════════════════════════════════════════

ENERGY_DRINK_STUDENTS: list[Persona] = [
    ("Chloe Bernard", 22, "Graduate Student", "Caffeine-Driven Graduate Student", "Late-night research, study sprints, cramming for exams."),
    ("Marcus Hwang", 19, "Engineering Undergrad", "Late-Night Coder", "Pulls all-nighters for problem sets and hackathons."),
    ("Priya Sharma", 24, "Med School Student", "Pre-Med Grinder", "Lives on caffeine through anatomy labs and board prep."),
    ("Jamal Williams", 20, "Business Major", "Group-Project Coordinator", "Juggles 5 projects, needs sustained focus."),
    ("Eun-ji Park", 21, "Architecture Student", "Studio All-Nighter", "Pulls 3 AM studios with looming model deadlines."),
    ("Tyler Brennan", 22, "Law Student", "Bar-Prep Grinder", "Outlines and case briefs into the small hours."),
    ("Aisha Hassan", 23, "PhD Candidate", "Dissertation Writer", "Writing chapters, needs 4-hour focus blocks."),
    ("Connor O'Brien", 19, "CS Undergrad", "Hackathon Regular", "Codes through weekend hackathons, scholarship-funded."),
    ("Diana Velasquez", 21, "Pre-Med", "Lab + Lecture Marathoner", "8 AM organic chem then 11 PM cell bio."),
    ("Raj Patel", 22, "MBA First-Year", "Case-Study Crusher", "Cases due daily, group meetings every night."),
    ("Sofia Castro", 20, "Biology Major", "Research Assistant", "Tracks cell cultures + 18-credit semester."),
    ("Trevor Lin", 24, "Med School Year 2", "STEP Studier", "300-hour USMLE prep cycle."),
    ("Nadia Khouri", 23, "Public Health Grad", "Thesis Researcher", "Field surveys + writing simultaneously."),
    ("Owen Kelleher", 21, "Engineering Junior", "Senior Project Lead", "Capstone team coordination + 18 units."),
    ("Maya Goldberg", 22, "Psych PhD First-Year", "Literature-Review Grinder", "Journal articles all weekend."),
]

ENERGY_DRINK_FITNESS: list[Persona] = [
    ("Brooke Stephens", 30, "Personal Trainer", "Fitness Optimizer", "Pre-workout boost for client sessions and personal lifts."),
    ("Marcus Johnson", 32, "CrossFit Coach", "Performance Athlete", "Tracks every macro, optimizes pre-WOD energy."),
    ("Diego Vargas", 28, "Powerlifter", "Strength Sport Athlete", "Heavy training blocks, needs caffeine timing."),
    ("Hiroshi Tanaka", 35, "Gym Owner", "Operator + Lifter", "Opens gym at 5 AM, trains between client sessions."),
    ("Ravi Subramanian", 30, "Marathon Runner", "Endurance Athlete", "Long runs Saturday, easy days during the week."),
    ("Elena Castellanos", 35, "Pilates Instructor", "Group Fitness Pro", "Teaches 6 classes a day, energy stacking required."),
    ("Beau Marchetti", 32, "Spin Instructor", "Cardio-Heavy Trainer", "Five 6 AM classes per week, needs morning kick."),
    ("Talia Rosenberg", 28, "Yoga + Strength", "Hybrid Athlete", "Pilates AM, weights PM, recovery focused."),
    ("Saanvi Krishnan", 26, "Bodybuilder Prep", "Competition Athlete", "Cutting phase, needs energy without calories."),
    ("Felix Aurelius", 33, "Olympic Lift Coach", "Technical Strength", "Coaches morning, lifts heavy at noon."),
    ("Naomi Ferguson", 40, "Triathlon Master", "Master's Athlete", "70.3 training, strict caffeine cycling."),
    ("Lucas Nakamura", 32, "BJJ Competitor", "Combat Sports Athlete", "Roll all morning, lift afternoon."),
    ("Adaeze Okafor", 27, "Strength Coach", "S&C Professional", "Programs for college teams, lifts at night."),
    ("Xavier Beaumont", 23, "Track Athlete", "Sprint Training", "Track sessions Tuesday/Thursday, weights other days."),
    ("Yara Khalil", 28, "Climbing Coach", "Endurance Climber", "Outdoor projects, weekly bouldering sessions."),
]

ENERGY_DRINK_NIGHT_SHIFT: list[Persona] = [
    ("Fatima Hassan", 28, "ICU Nurse", "Exhausted Medical Professional", "Three 12-hour night shifts a week, caffeine essential."),
    ("Marcus Pierce", 35, "ER Doctor", "Night-Shift Physician", "Trauma center overnights, sleep cycle inverted."),
    ("Jonas Albrecht", 42, "Police Officer", "Overnight Patrol", "Shift work for 15 years, sleep is structured caffeine."),
    ("Reginald Cross", 45, "Firefighter", "24-Hour Shift Worker", "On-call all night, needs alertness on demand."),
    ("Linnea Eriksson", 38, "Surgical Nurse", "OR Night Crew", "Emergency surgery calls, unpredictable hours."),
    ("Otis Reynolds", 50, "Long-Haul Trucker", "Cross-Country Driver", "16-hour drives, alertness is safety."),
    ("Camille Dupree", 32, "Hotel Concierge", "Overnight Hospitality", "Front desk 11 PM to 7 AM."),
    ("Trevor Bishop", 35, "IT Operations", "On-Call Systems Engineer", "Production support overnights."),
    ("Robert Chen", 40, "Air Traffic Controller", "Night-Shift Aviation", "High-stakes, must stay sharp."),
    ("Elena Voss", 33, "Hospital Pharmacist", "Night-Shift Pharmacy", "Verifying meds on graveyard shift."),
    ("Asha Mehta", 28, "Paramedic", "EMS Overnight", "Calls all night, naps between."),
    ("Theo Whitaker", 38, "Bartender", "Late-Night Service", "Closes the bar, energy management is the job."),
    ("Naoki Yamamoto", 35, "Casino Dealer", "Overnight Gaming", "Stand for 8 hours, focused attention."),
    ("Marisol Ortega", 30, "Rideshare Driver", "Late-Night Driver", "Bar rush hours, drive until 4 AM."),
    ("Devon Sutherland", 42, "Security Supervisor", "Building Security", "Walks the floor every hour, needs alertness."),
]

ENERGY_DRINK_GAMERS: list[Persona] = [
    ("Kai Holloway", 22, "Streamer", "Twitch Streamer", "Streams 6+ hours, viewer-energy correlation."),
    ("Ava Thompson", 24, "Esports Pro", "Competitive Gamer", "Tournament prep, reflex-dependent role."),
    ("Elijah Foster", 22, "Content Creator", "YouTube + Twitch", "Records gameplay, edits late, repeats."),
    ("Soren Bjorklund", 25, "Game Designer", "Indie Dev", "Codes evenings after day job."),
    ("Niamh Doyle", 23, "Pro Gamer", "Apex Predator", "Ranked grind sessions, stim-stack."),
    ("Lucia Romano", 26, "Streamer + Cosplayer", "Variety Streamer", "Long-format content, 4-hour minimum."),
    ("Zara Ahmed", 24, "Esports Coach", "Team Strategist", "VOD review marathons."),
    ("Jamie Reyes", 21, "College Esports Player", "Scholarship Athlete", "Practices 4 hours after class."),
    ("Felix Aurelius", 26, "Speedrunner", "Record Holder", "Hours of attempts in single sessions."),
    ("Talia Bergman", 22, "Twitch Variety", "Just-Chatting Streamer", "Engages chat for 5+ hours straight."),
    ("Owen Sinclair", 25, "FGC Competitor", "Fighting Game Pro", "Tournament travel, jet lag energy."),
    ("Camille Faure", 24, "MMO Raider", "Hardcore Endgame", "10 PM raids Tuesday/Thursday."),
    ("Diego Vargas Jr", 22, "Console FPS Pro", "Controller Pro", "Scrims and tournaments back-to-back."),
    ("Mei-ling Wu", 23, "VTuber", "Animated Streamer", "8-hour streams in costume."),
    ("Beau Carter", 20, "College Streamer", "Dorm-Room Content", "Streams between classes."),
]


# ═══════════════════════════════════════════════════════════════════════
# HYDRATION PERSONAS (3 sub-pools × 15 = 45 personas)
# ═══════════════════════════════════════════════════════════════════════

HYDRATION_ATHLETES: list[Persona] = [
    ("Brooke Stephens", 30, "Marathon Runner", "Endurance Athlete", "Long runs need electrolyte replacement."),
    ("Marcus Johnson", 32, "Triathlete", "Triathlon Competitor", "Multi-hour brick workouts in heat."),
    ("Naomi Ferguson", 40, "Master's Cyclist", "Long-Distance Cyclist", "Centuries every weekend."),
    ("Diego Vargas", 28, "BJJ Fighter", "Combat Sports", "Sweats heavily during 2-hour rolls."),
    ("Lucas Nakamura", 32, "Hiker", "Multi-Day Backpacker", "Carries hydration for trail days."),
    ("Saanvi Krishnan", 26, "Bodybuilder", "Cutting-Phase Athlete", "Sodium/potassium balance critical."),
    ("Hiroshi Tanaka", 35, "Tennis Player", "Match-Day Athlete", "3-set matches in summer heat."),
    ("Ravi Subramanian", 30, "Ultra Runner", "Ultramarathoner", "50K+ races, salt management."),
    ("Talia Rosenberg", 28, "Soccer Player", "Adult League", "90-minute games twice weekly."),
    ("Beau Marchetti", 32, "CrossFit Athlete", "Hyrox Competitor", "Workout heat, sweat-rate aware."),
    ("Felix Aurelius", 33, "Olympic Weightlifter", "Strength Athlete", "Multi-hour training, hydration logged."),
    ("Adaeze Okafor", 27, "Sprinter", "Track Athlete", "Track session water bottles always."),
    ("Xavier Beaumont", 23, "Lacrosse Player", "Field Sport Athlete", "Practices in summer humidity."),
    ("Yara Khalil", 28, "Climber", "Multi-Pitch Climber", "Long days on the rock."),
    ("Owen Kelleher", 25, "Rugby Player", "Contact Sport Athlete", "Training-camp dehydration risk."),
]

HYDRATION_WELLNESS_PROFESSIONALS: list[Persona] = [
    ("Mei Lin Zhao", 28, "Marketing Manager", "Wellness-Conscious Professional", "Tracks water intake on app."),
    ("Priya Patel", 30, "Biotech PM", "Health Optimizer", "Reads about electrolyte balance."),
    ("Maya Goldberg", 27, "UX Designer", "Self-Care Routine", "Morning hydration ritual."),
    ("Theo Whitaker", 38, "Tech Lead", "Performance-Conscious Pro", "Optimizes work cognition."),
    ("Brigitte Laurent", 38, "Wellness Coach", "Premium Health Buyer", "Recommends to clients."),
    ("Talia Bergman", 32, "Therapist", "Mind-Body Aware", "Hydration and mental clarity link."),
    ("Camille Dupree", 26, "Yoga Studio Manager", "Studio Lifestyle", "Class teachers go through 2L daily."),
    ("Ravi Subramanian", 30, "Investment Analyst", "Optimization-Focused", "Stack with morning routine."),
    ("Adaeze Okafor", 27, "PR Director", "Wellness-Forward Pro", "Travel days, plane dehydration."),
    ("Saanvi Krishnan", 26, "Health Coach", "Biohacker", "Tracks hydration markers."),
    ("Owen Sinclair", 30, "Architect", "Mid-Career Pro", "Hydration as productivity tool."),
    ("Niamh Doyle", 29, "Nutritionist", "Wellness Practitioner", "Educates clients on hydration."),
    ("Jonas Albrecht", 40, "Executive", "C-Suite Health-Focused", "Travel + meetings stamina."),
    ("Linnea Eriksson", 38, "RN Wellness Nurse", "Health-First Professional", "Promotes electrolytes to patients."),
    ("Camille Faure", 30, "Acupuncturist", "Holistic Practitioner", "Recommends to acu clients."),
]

HYDRATION_BUSY_PARENTS: list[Persona] = [
    ("Isabella Santos", 32, "Mother of Two", "Time-Pressed Parent", "Hydration before school drop-off."),
    ("Carlos Mendoza", 40, "Father of Three", "Suburban Dad", "Whole family drinks electrolytes."),
    ("Elena Castellanos", 35, "Working Mom", "Dual-Income Parent", "Soccer practice survival."),
    ("Ines Ribeiro", 38, "Mother of Twins", "Twin Mom", "Constant motion, constant hydration."),
    ("Naomi Ferguson", 40, "Mom of Teenagers", "Sports Parent", "Stocks teen athlete fridge."),
    ("Marisol Ortega", 30, "Single Mom", "Solo Parent", "Buys for kids first, herself second."),
    ("Trevor Bishop", 35, "Stay-at-Home Dad", "Primary Caregiver", "Park days, summer survival."),
    ("Yara Khalil", 28, "First-Time Mom", "New Parent", "Postpartum hydration."),
    ("Reginald Cross", 45, "Empty-Nest Dad", "Active Older Parent", "Adult kids visit on holidays."),
    ("Camille Dupree", 32, "Mom + WFH", "Hybrid Parent", "Home-school + work."),
    ("Asha Mehta", 28, "Pregnant Mom", "Expecting", "OBGYN recommended."),
    ("Devon Sutherland", 42, "Coaching Dad", "Sports-Coach Parent", "Soccer + baseball coach."),
    ("Theo Whitaker", 38, "Travel Dad", "Frequent Flyer Parent", "Plane trips with kids."),
    ("Beau Marchetti", 32, "Active Dad", "Outdoor Family", "Hiking with toddlers."),
    ("Soren Bjorklund", 30, "Dad of Newborn", "Sleep-Deprived Parent", "3 AM feedings, dehydration."),
]


# ═══════════════════════════════════════════════════════════════════════
# SUPPLEMENTS PERSONAS (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

SUPPLEMENTS_GYM_GOERS: list[Persona] = [
    ("Marcus Johnson", 32, "Personal Trainer", "Gym Regular", "Stacks pre/intra/post for clients and self."),
    ("Diego Vargas", 28, "Powerlifter", "Strength Athlete", "Creatine-loaded year-round."),
    ("Hiroshi Tanaka", 35, "Gym Owner", "Operator-Athlete", "Tries supplements for clients."),
    ("Saanvi Krishnan", 26, "Competitor Bodybuilder", "Show-Prep Athlete", "Strict timing on cycle."),
    ("Felix Aurelius", 33, "Olympic Lifter", "Technical Strength", "Caffeine + creatine stack."),
    ("Beau Marchetti", 32, "CrossFit Athlete", "Functional Fitness", "Daily greens + protein."),
    ("Brooke Stephens", 30, "Online Trainer", "Influencer Trainer", "Affiliate codes for everything."),
    ("Adaeze Okafor", 27, "Strength Coach", "S&C Pro", "Tests products before recommending."),
    ("Owen Kelleher", 25, "Rugby Player", "Team Sport Athlete", "Recovery-stack focused."),
    ("Lucas Nakamura", 32, "BJJ Competitor", "Grappler", "Joint health for older grappler."),
    ("Talia Rosenberg", 28, "Hybrid Athlete", "Multi-Modal", "Pilates + lifting protocol."),
    ("Xavier Beaumont", 23, "Track Athlete", "Sprinter", "Beta-alanine for explosive."),
    ("Ravi Subramanian", 30, "Endurance Lifter", "Hybrid Athlete", "Hyrox plus marathon."),
    ("Naomi Ferguson", 40, "Master's Lifter", "Older Strength Athlete", "Joints first, performance second."),
    ("Yara Khalil", 28, "Climber", "Strength-to-Weight", "Lean-mass focus."),
]

SUPPLEMENTS_OPTIMIZERS: list[Persona] = [
    ("Hiroshi Tanaka", 35, "Biotech PM", "Hardcore Optimizer", "Tracks bloodwork and HRV."),
    ("Ravi Subramanian", 30, "Quant Analyst", "Data-Driven Health", "Spreadsheets every supplement."),
    ("Theo Whitaker", 38, "Tech Lead", "Performance Pro", "Stack designed for cognition."),
    ("Priya Patel", 30, "Med Researcher", "Evidence-Based Stacker", "Reads PubMed studies."),
    ("Felix Aurelius", 33, "Engineer", "Quantified Self", "Wearable + supplement correlations."),
    ("Maya Goldberg", 27, "Psychiatry Resident", "Brain Health", "Nootropic curious."),
    ("Diego Vargas", 28, "Software Engineer", "Code + Bio", "Builds personal health stack."),
    ("Talia Rosenberg", 28, "Health Coach", "Functional Practitioner", "Recommends evidence-backed."),
    ("Adaeze Okafor", 27, "PhD Biology", "Researcher", "Reads abstract before buying."),
    ("Owen Sinclair", 30, "Investment Pro", "Burnout-Adjacent Optimizer", "Wants edge, sustainable."),
    ("Saanvi Krishnan", 26, "Biotech Founder", "Startup Founder", "Optimizes for intensity."),
    ("Niamh Doyle", 29, "Pharmacist", "Medication-Aware Optimizer", "Knows interactions."),
    ("Brigitte Laurent", 38, "Wellness Practitioner", "Functional Medicine", "Targeted supplementation."),
    ("Soren Bjorklund", 30, "Bioinformatics", "DNA-Tested Stacker", "MTHFR-aware."),
    ("Camille Faure", 30, "Acupuncturist", "Holistic Optimizer", "TCM-informed stacking."),
]

SUPPLEMENTS_GENERAL_HEALTH: list[Persona] = [
    ("Linnea Eriksson", 38, "RN", "Health-First Professional", "Recommends evidence-backed."),
    ("Carlos Mendoza", 40, "Office Worker", "Mid-Life Health-Conscious", "Multivitamin daily."),
    ("Elena Castellanos", 35, "Teacher", "Wellness Maintainer", "Routine multivit + D3."),
    ("Reginald Cross", 45, "Manager", "Heart-Health Aware", "Omega-3 daily."),
    ("Naomi Ferguson", 40, "Working Mom", "Family Health-Focused", "Whole family takes."),
    ("Jonas Albrecht", 42, "Executive", "Mid-Career Health", "Stress-management stack."),
    ("Marisol Ortega", 30, "Single Mom", "Practical Buyer", "Cost-conscious, basics only."),
    ("Otis Reynolds", 50, "Pre-Retiree", "Aging Adult", "Joints, heart, vision."),
    ("Camille Dupree", 32, "Marketing", "General Wellness", "Daily routine vitamins."),
    ("Talia Bergman", 32, "Counselor", "Mood-Support Buyer", "Mental wellness focus."),
    ("Trevor Bishop", 35, "IT Manager", "Routine User", "Same supps for years."),
    ("Isabella Santos", 32, "Mom", "Family Buyer", "Buys for husband and self."),
    ("Devon Sutherland", 42, "Sales Manager", "Stamina Buyer", "Travel-heavy job."),
    ("Asha Mehta", 28, "Pregnant", "Prenatal Buyer", "OB-recommended."),
    ("Beau Marchetti", 32, "Active Dad", "Family Health", "Whole-family supp box."),
]


# ═══════════════════════════════════════════════════════════════════════
# SKINCARE PERSONAS (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

SKINCARE_ENTHUSIASTS: list[Persona] = [
    ("Sophia Russo", 28, "Beauty Editor", "Skincare Maximalist", "10-step routine, ingredient-aware."),
    ("Maya Goldberg", 27, "Beauty Influencer", "Trend-Aware Buyer", "Tries new launches monthly."),
    ("Priya Patel", 30, "Dermatology PA", "Pro-Adjacent Buyer", "Knows actives by molecular weight."),
    ("Camille Dupree", 32, "Aesthetician", "Industry Insider", "Recommends to clients."),
    ("Talia Rosenberg", 28, "Makeup Artist", "Pro Beauty Pro", "Tests on shoots."),
    ("Eun-ji Park", 25, "K-Beauty Convert", "K-Beauty Loyalist", "Imports Korean brands."),
    ("Brigitte Laurent", 38, "Wellness Coach", "Holistic Skincare", "Clean beauty advocate."),
    ("Yara Khalil", 28, "Personal Stylist", "Aesthetic-First Buyer", "Routine = ritual."),
    ("Adaeze Okafor", 27, "PR Director", "Industry-Aware", "Gets PR samples constantly."),
    ("Saanvi Krishnan", 26, "Beauty Founder", "Indie-Brand Tester", "Knows formulator quality."),
    ("Niamh Doyle", 29, "Pharmacist", "Active-Ingredient Aware", "Verifies % concentrations."),
    ("Camille Faure", 30, "Esthetician Owner", "Spa Owner", "Educated retail."),
    ("Asha Mehta", 28, "Beauty Tech", "Indie Founder", "Reformulates favorites."),
    ("Lucia Romano", 26, "Beauty Vlogger", "Reviewer", "Posts long-form analysis."),
    ("Mei-ling Wu", 23, "Aesthetic-Forward Buyer", "Visual Identity", "Skincare as identity."),
]

SKINCARE_ACNE_PRONE: list[Persona] = [
    ("Ava Thompson", 24, "Marketing Coordinator", "Adult-Acne Sufferer", "Hormonal breakouts persistent."),
    ("Tyler Brennan", 22, "Recent Grad", "Cystic-Acne History", "Tried Accutane, maintaining."),
    ("Jamie Reyes", 21, "College Junior", "Sensitive + Acne-Prone", "Reactive skin, cautious buyer."),
    ("Diana Velasquez", 21, "Pre-Med", "Stress Acne", "Med school stress = breakouts."),
    ("Connor O'Brien", 19, "Undergrad", "Teen-Acne Adult", "Lingering teen acne at 19."),
    ("Nadia Khouri", 23, "Public Health", "PCOS Acne", "Medical-cause acne."),
    ("Felix Aurelius", 26, "Engineer", "Adult-Onset Male Acne", "Stress + travel breakouts."),
    ("Talia Bergman", 22, "Recent Grad", "Inflammatory Skin", "Rosacea-acne overlap."),
    ("Maya Patel", 25, "Marketing Junior", "PIE/PIH Battle", "Post-acne hyperpigmentation."),
    ("Camille Faure", 28, "Aesthetician", "Cysticc-Prone Pro", "Treats own + clients."),
    ("Owen Sinclair", 25, "Architect", "Beard-Area Breakouts", "Razor-bump acne."),
    ("Adaeze Okafor", 27, "PR", "Stress Acne Pro", "Travel breakouts."),
    ("Soren Bjorklund", 30, "Bioinformatics", "Late-Onset Acne", "Adult-onset hormonal."),
    ("Eun-ji Park", 24, "Designer", "K-Acne Routine", "Korean acne strategy."),
    ("Ines Ribeiro", 27, "Mother", "Postpartum Acne", "Hormonal post-baby."),
]

SKINCARE_ANTI_AGING: list[Persona] = [
    ("Brigitte Laurent", 38, "Brand Director", "Anti-Aging Buyer", "Ingredient-led routine."),
    ("Naomi Ferguson", 40, "Marketing VP", "Pro-Aging Optimizer", "Retinol + peptides."),
    ("Reginald Cross", 45, "Executive", "Mature Male Skincare", "Recently started."),
    ("Linnea Eriksson", 38, "RN", "Evidence-Based Aging", "Studies the studies."),
    ("Jonas Albrecht", 42, "C-Suite", "Late-Adopter Male", "Wife-recommended."),
    ("Otis Reynolds", 50, "Pre-Retiree", "Sun-Damage Repair", "Decades of sun, fixing now."),
    ("Theo Whitaker", 38, "Tech Lead", "Tech-Career Skincare", "Screen time + late nights."),
    ("Camille Dupree", 32, "Yoga Manager", "Early Anti-Aging", "Prevention mindset."),
    ("Brigitte Faure", 35, "Architect", "Career-Aging Pro", "Investing in skin."),
    ("Devon Sutherland", 42, "Sales", "Mid-40s Routine", "Wife introduced."),
    ("Trevor Bishop", 35, "IT Manager", "Pre-Aging Adult", "Fine-line concerns."),
    ("Marisol Ortega", 38, "Mom", "Post-Pregnancy Aging", "Skin changed after kids."),
    ("Robert Chen", 40, "Executive", "Travel-Aging", "Frequent flyer skin damage."),
    ("Elena Voss", 33, "Pharmacist", "Active-Ingredient Aware", "Knows the science."),
    ("Niamh Doyle", 39, "Nutritionist", "Inside-Out Aging", "Diet + topical."),
]


# ═══════════════════════════════════════════════════════════════════════
# PERSONAL CARE PERSONAS (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

PERSONAL_CARE_MEN_GROOMING: list[Persona] = [
    ("Marcus Johnson", 32, "Trainer", "Beard-Maintenance Pro", "Daily beard routine."),
    ("Diego Vargas", 28, "Lawyer", "Sharp-Look Pro", "Clean-shaven daily."),
    ("Hiroshi Tanaka", 35, "Owner", "Mature Beard", "Years-long beard maintainer."),
    ("Felix Aurelius", 33, "Engineer", "Beard Newcomer", "First-year beard growing."),
    ("Beau Marchetti", 32, "Architect", "Designer-Stubble Pro", "Maintains 5-day shadow."),
    ("Owen Sinclair", 30, "Designer", "Aesthetic-First Male", "Skincare-curious male."),
    ("Theo Whitaker", 38, "Executive", "Corporate Groomer", "Polished professional look."),
    ("Reginald Cross", 45, "Manager", "Salt-and-Pepper Male", "Older grooming customer."),
    ("Lucas Nakamura", 32, "Tech", "Asian-Hair Beard", "Sparse-beard challenge."),
    ("Trevor Bishop", 35, "IT Manager", "Practical Male", "Minimal routine, effective."),
    ("Devon Sutherland", 42, "Sales", "Travel-Pro Groomer", "On-the-road grooming."),
    ("Carlos Mendoza", 40, "Manager", "Family-Man Groomer", "Quick effective routine."),
    ("Robert Chen", 40, "Executive", "Investment-Skincare Male", "Premium products."),
    ("Jonas Albrecht", 42, "C-Suite", "Late-Adopter Skincare Male", "Started in 40s."),
    ("Otis Reynolds", 50, "Pre-Retiree", "Mature Male Self-Care", "New routine recently."),
]

PERSONAL_CARE_RAZOR_SUBSCRIBERS: list[Persona] = [
    ("Marcus Johnson", 32, "Trainer", "Subscription-Box Loyal", "Auto-delivery believer."),
    ("Diego Vargas", 28, "Lawyer", "Daily-Shaver Pro", "Sharp blade obsessive."),
    ("Felix Aurelius", 33, "Engineer", "DIY-Curious", "Considered safety razor."),
    ("Hiroshi Tanaka", 35, "Owner", "Subscription Skeptic", "Tried, returned to retail."),
    ("Beau Marchetti", 32, "Architect", "Premium-Razor Buyer", "Pays for quality blade."),
    ("Owen Sinclair", 30, "Designer", "Aesthetic-Box Buyer", "Likes the unboxing."),
    ("Theo Whitaker", 38, "Tech Lead", "Convenience Buyer", "Subscription = no thinking."),
    ("Reginald Cross", 45, "Manager", "Deal-Hunter Subscriber", "Switched for promo."),
    ("Trevor Bishop", 35, "IT Manager", "Practical Subscriber", "Auto-delivery years."),
    ("Lucas Nakamura", 32, "Tech", "Sensitive-Skin Subscriber", "Specific blade need."),
    ("Devon Sutherland", 42, "Sales", "Travel Pack Buyer", "On-road shave kit."),
    ("Carlos Mendoza", 40, "Manager", "Family Multi-Subscriber", "Family of male shavers."),
    ("Robert Chen", 40, "Executive", "Premium-Service Buyer", "Concierge expectations."),
    ("Connor O'Brien", 22, "Recent Grad", "First-Time Subscriber", "Just signed up."),
    ("Tyler Brennan", 22, "New Grad", "College-Era Subscriber", "Started in college."),
]

PERSONAL_CARE_BEARD: list[Persona] = [
    ("Marcus Johnson", 32, "Trainer", "Year-2 Beard", "Growing in commitment."),
    ("Hiroshi Tanaka", 35, "Owner", "Long-Beard Veteran", "5+ year beard."),
    ("Felix Aurelius", 33, "Engineer", "Beardsman Hobbyist", "Beard-care community."),
    ("Beau Marchetti", 32, "Architect", "Curated-Beard Pro", "Shapes carefully."),
    ("Diego Vargas", 28, "Lawyer", "Trial-Beard Buyer", "First-year experimenting."),
    ("Reginald Cross", 45, "Manager", "Mature-Beard Buyer", "Maintains gray beard."),
    ("Owen Sinclair", 30, "Designer", "Designer-Stubble Pro", "Manages 5-day."),
    ("Lucas Nakamura", 32, "Tech", "Sparse-Beard Customer", "Patchy challenge."),
    ("Theo Whitaker", 38, "Executive", "Corporate Beard", "Maintains for office."),
    ("Trevor Bishop", 35, "IT Manager", "Casual Beard Pro", "Low-maintenance grower."),
    ("Devon Sutherland", 42, "Sales", "Salesmen Beard", "Customer-facing groomed."),
    ("Carlos Mendoza", 40, "Manager", "Family Beard", "Father-figure beard."),
    ("Robert Chen", 40, "Executive", "Premium-Beard Buyer", "Top-tier products."),
    ("Jonas Albrecht", 42, "C-Suite", "Late-Beard Adopter", "Started in 40s."),
    ("Otis Reynolds", 50, "Pre-Retiree", "Retirement-Beard Pro", "Just-letting-it-grow phase."),
]


# ═══════════════════════════════════════════════════════════════════════
# NA BEER PERSONAS (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

NA_BEER_SOBER_CURIOUS: list[Persona] = [
    ("Brooke Stephens", 30, "Trainer", "Dry-January Veteran", "Annual dry months."),
    ("Maya Goldberg", 27, "Designer", "Mindful Drinker", "Cutting back, not stopping."),
    ("Theo Whitaker", 38, "Tech Lead", "Performance-Mindset", "Sleep + recovery."),
    ("Talia Rosenberg", 28, "Therapist", "Wellness Lifestyle", "Mental health priority."),
    ("Jonas Albrecht", 42, "Executive", "Health-Awareness Adult", "Reduced after 40."),
    ("Brigitte Laurent", 38, "Coach", "Holistic Coach", "Promotes to clients."),
    ("Diego Vargas", 28, "Lawyer", "Career-Focused Sober", "No-hangover priority."),
    ("Felix Aurelius", 33, "Engineer", "Optimization Sober", "Better sleep + cognition."),
    ("Lucas Nakamura", 32, "Tech", "Early-30s Reset", "Cut back gradually."),
    ("Saanvi Krishnan", 26, "Founder", "Founder-Life Sober", "Startup demands."),
    ("Niamh Doyle", 29, "Pharmacist", "Medication-Aware Sober", "Health profession."),
    ("Owen Sinclair", 30, "Architect", "Quiet Quitter", "Slowly stopped."),
    ("Adaeze Okafor", 27, "PR", "Career + Health Sober", "Industry pressure."),
    ("Camille Dupree", 32, "Yoga Manager", "Yoga-Life Sober", "Studio lifestyle."),
    ("Reginald Cross", 45, "Manager", "Mid-Life Sober", "Started cutting at 40."),
]

NA_BEER_CRAFT_BEER_FANS: list[Persona] = [
    ("Marcus Johnson", 32, "Trainer", "Craft-Beer Fitness", "Loves taste, not effects."),
    ("Diego Vargas", 28, "Lawyer", "Craft-Connoisseur", "Knows IPAs deeply."),
    ("Hiroshi Tanaka", 35, "Owner", "Brewery-Tour Veteran", "Visited 100+ breweries."),
    ("Beau Marchetti", 32, "Architect", "Style-Specific Beer Fan", "Wee Heavy enthusiast."),
    ("Owen Sinclair", 30, "Designer", "Aesthetic-Label Buyer", "Buys for can art."),
    ("Felix Aurelius", 33, "Engineer", "Hop-Forward Fan", "IPAs only."),
    ("Theo Whitaker", 38, "Tech Lead", "Saturday-Night Sipper", "Quality over quantity."),
    ("Reginald Cross", 45, "Manager", "Beer-Club Member", "Monthly delivery."),
    ("Lucas Nakamura", 32, "Tech", "Asian-Brewery Fan", "Sapporo + craft."),
    ("Trevor Bishop", 35, "IT Manager", "Local-Brewery Loyal", "Supports neighborhood."),
    ("Devon Sutherland", 42, "Sales", "Travel-Brewery Hunter", "Trips for breweries."),
    ("Carlos Mendoza", 40, "Manager", "Mexican-Lager Loyalist", "Modelo + craft hybrid."),
    ("Robert Chen", 40, "Executive", "Premium-Imports Fan", "European-only."),
    ("Jonas Albrecht", 42, "C-Suite", "Vintage-Beer Cellarer", "Ages bottles."),
    ("Otis Reynolds", 50, "Pre-Retiree", "Lifelong Beer Fan", "50+ years drinking."),
]

NA_BEER_FITNESS: list[Persona] = [
    ("Brooke Stephens", 30, "Trainer", "Athlete-First Sober", "Performance-driven."),
    ("Marcus Johnson", 32, "CrossFit Coach", "Fitness Sober", "Training-cycle aware."),
    ("Diego Vargas", 28, "Powerlifter", "Sports-Sober Athlete", "Comp prep dry."),
    ("Saanvi Krishnan", 26, "Bodybuilder", "Show-Prep Sober", "Cutting-phase NA."),
    ("Hiroshi Tanaka", 35, "Gym Owner", "Industry-Sober Pro", "Owns gym, drinks NA."),
    ("Felix Aurelius", 33, "Olympic Lifter", "Strength Sober", "Recovery first."),
    ("Beau Marchetti", 32, "CrossFit Athlete", "Hyrox-Sober", "Training year-round."),
    ("Adaeze Okafor", 27, "S&C Coach", "Pro-Coach Sober", "Athletes notice."),
    ("Lucas Nakamura", 32, "BJJ Fighter", "Comp-Sober Grappler", "Year-round prep."),
    ("Talia Rosenberg", 28, "Hybrid Athlete", "Multi-Sport Sober", "Always in season."),
    ("Xavier Beaumont", 23, "Track Athlete", "NCAA-Era Sober", "College athlete."),
    ("Yara Khalil", 28, "Climber", "Outdoor-Sober", "Climbing trips, NA."),
    ("Naomi Ferguson", 40, "Master's Triathlete", "Master-Athlete Sober", "Training schedule."),
    ("Ravi Subramanian", 30, "Ultra Runner", "Endurance Sober", "Long-run respecter."),
    ("Owen Kelleher", 25, "Rugby Player", "Team-Sport Sober", "In-season abstinent."),
]


# ═══════════════════════════════════════════════════════════════════════
# COFFEE ALTERNATIVE PERSONAS (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

COFFEE_ALT_WELLNESS: list[Persona] = [
    ("Maya Goldberg", 27, "Designer", "Wellness-Curious", "Tries new wellness products."),
    ("Brigitte Laurent", 38, "Coach", "Holistic Coach", "Recommends mushroom adaptogens."),
    ("Talia Rosenberg", 28, "Therapist", "Mind-Body Aware", "Adaptogen experimenter."),
    ("Priya Patel", 30, "Researcher", "Ingredient-Forward", "Reads about cordyceps."),
    ("Camille Dupree", 32, "Yoga Manager", "Studio-Life", "Promotes to students."),
    ("Saanvi Krishnan", 26, "Founder", "Wellness-Founder", "Stack with morning routine."),
    ("Niamh Doyle", 29, "Pharmacist", "Pharmacist-Curious", "Vetting science."),
    ("Adaeze Okafor", 27, "PR", "Wellness-Industry Pro", "Tries client products."),
    ("Camille Faure", 30, "Acupuncturist", "TCM Practitioner", "Lions-mane long history."),
    ("Brigitte Faure", 35, "Architect", "Mid-Career Wellness", "Coffee-replacement journey."),
    ("Theo Whitaker", 38, "Tech Lead", "Cognition Buyer", "Brain-focus product."),
    ("Soren Bjorklund", 30, "Bioinformatics", "Data-Driven Wellness", "Tracks effects."),
    ("Felix Aurelius", 33, "Engineer", "Optimization Curious", "Trial-and-data approach."),
    ("Yara Khalil", 28, "Stylist", "Lifestyle Buyer", "Aesthetic + function."),
    ("Linnea Eriksson", 38, "RN", "Health-First Buyer", "Patient-recommendation aware."),
]

COFFEE_ALT_COFFEE_REDUCERS: list[Persona] = [
    ("Theo Whitaker", 38, "Tech Lead", "Caffeine-Cutter", "Reducing daily intake."),
    ("Maya Goldberg", 27, "Designer", "Anxiety-Cutter", "Coffee = anxiety, switching."),
    ("Diego Vargas", 28, "Lawyer", "Sleep-First Cutter", "Better sleep, less caffeine."),
    ("Reginald Cross", 45, "Manager", "Heart-Health Cutter", "Doctor recommended."),
    ("Brigitte Laurent", 38, "Coach", "Coach-Self-Cutting", "Demonstrates to clients."),
    ("Saanvi Krishnan", 26, "Founder", "Burnout-Recovery", "Cutting after burnout."),
    ("Talia Rosenberg", 28, "Therapist", "Anxiety-Sensitive Cutter", "Mental health priority."),
    ("Camille Dupree", 32, "Yoga Manager", "Yoga-Life Cutter", "Studio lifestyle."),
    ("Felix Aurelius", 33, "Engineer", "Performance-Reducer", "Notices afternoon crash."),
    ("Owen Sinclair", 30, "Architect", "Heartburn Cutter", "GI-driven reduction."),
    ("Hiroshi Tanaka", 35, "Owner", "Reflux-Cutter", "Acid-reflux issues."),
    ("Trevor Bishop", 35, "IT Manager", "40s-Reducer", "Mid-life caffeine cut."),
    ("Adaeze Okafor", 27, "PR", "Cycle-Aware Cutter", "Hormonal sensitivity."),
    ("Lucas Nakamura", 32, "Tech", "30s-Realization Cutter", "Body changed."),
    ("Carlos Mendoza", 40, "Manager", "Family-Cycle Cutter", "Modeled for kids."),
]

COFFEE_ALT_MUSHROOM_CURIOUS: list[Persona] = [
    ("Maya Goldberg", 27, "Designer", "Functional-Mushroom Curious", "Saw on TikTok."),
    ("Felix Aurelius", 33, "Engineer", "Nootropic-Curious", "Reads about cognition."),
    ("Saanvi Krishnan", 26, "Biotech", "Lions-Mane Researcher", "Read the studies."),
    ("Theo Whitaker", 38, "Tech Lead", "Cognition-First Buyer", "Optimizing focus."),
    ("Talia Rosenberg", 28, "Therapist", "Adaptogen-Curious", "Stress management."),
    ("Brigitte Laurent", 38, "Coach", "Functional-Food Educator", "Educates clients."),
    ("Priya Patel", 30, "Researcher", "Bioactive-Curious", "Wants research-grade."),
    ("Niamh Doyle", 29, "Pharmacist", "Skeptical Curious", "Verifying claims."),
    ("Camille Faure", 30, "Acupuncturist", "TCM-Knowledgeable", "Ancient roots."),
    ("Lucia Romano", 26, "Vlogger", "Trend-Forward Buyer", "Reviews for audience."),
    ("Yara Khalil", 28, "Stylist", "Lifestyle-First Buyer", "Aesthetic of mushroom."),
    ("Owen Sinclair", 30, "Designer", "Daily-Routine Buyer", "Replacing daily coffee."),
    ("Adaeze Okafor", 27, "PR", "Industry-Curious Pro", "Knows the brands."),
    ("Soren Bjorklund", 30, "Bioinformatics", "Data-Heavy Buyer", "Tracks effects."),
    ("Eun-ji Park", 25, "Designer", "K-Wellness Bridge", "Korean adaptogen culture."),
]


# ═══════════════════════════════════════════════════════════════════════
# MATTRESS PERSONAS (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

MATTRESS_FIRST_BUYERS: list[Persona] = [
    ("Tyler Brennan", 24, "New Grad", "First-Apartment Buyer", "First adult mattress."),
    ("Connor O'Brien", 23, "Junior Dev", "Post-College Upgrader", "Replacing dorm mattress."),
    ("Diana Velasquez", 25, "Med Resident", "Resident-Salary Buyer", "First real bed."),
    ("Eun-ji Park", 24, "Designer", "Studio-Apartment Buyer", "First city move."),
    ("Jamie Reyes", 22, "Recent Grad", "Roommate-to-Self Buyer", "Out of roommate situation."),
    ("Asha Mehta", 25, "Junior Pro", "First Solo Place", "First non-shared bed."),
    ("Trevor Lin", 26, "Med Student", "Med-School Move", "Better-than-dorm upgrade."),
    ("Maya Patel", 25, "Marketing Junior", "Quarter-Life Upgrade", "Adulting purchase."),
    ("Nadia Khouri", 23, "PhD Student", "Grad-Student Upgrade", "First new mattress."),
    ("Diego Vargas Jr", 24, "Tech Junior", "First-Pro-Salary Buyer", "Earned-it purchase."),
    ("Talia Bergman", 22, "New Grad", "Out-of-State Move", "New city, new everything."),
    ("Camille Faure", 28, "Acupuncturist", "Practitioner Upgrader", "Spine health priority."),
    ("Owen Kelleher", 25, "Rugby Player", "Athlete First-Mattress", "Recovery priority."),
    ("Soren Bjorklund", 30, "Engineer", "Late-First Buyer", "Finally upgrading."),
    ("Lucia Romano", 26, "Content Creator", "Creator-First Adult Buy", "Visible bedroom on stream."),
]

MATTRESS_UPGRADERS: list[Persona] = [
    ("Carlos Mendoza", 40, "Manager", "Family-Bed Upgrader", "10-year mattress replace."),
    ("Elena Castellanos", 35, "Teacher", "Mid-Life Upgrade", "Better-quality buyer."),
    ("Reginald Cross", 45, "Manager", "Empty-Nest Upgrade", "Investing post-kids."),
    ("Naomi Ferguson", 40, "Marketing VP", "Sleep-Quality Buyer", "Performance-life link."),
    ("Theo Whitaker", 38, "Tech Lead", "WFH-Upgrade Buyer", "Home office + bed time."),
    ("Brigitte Laurent", 38, "Coach", "Wellness-Lifestyle Upgrade", "Holistic health."),
    ("Jonas Albrecht", 42, "Executive", "Quality-of-Life Upgrade", "Affording the best."),
    ("Robert Chen", 40, "Executive", "Premium-Tier Buyer", "Top-of-line only."),
    ("Trevor Bishop", 35, "IT Manager", "Routine Upgrader", "Every 8 years."),
    ("Devon Sutherland", 42, "Sales", "Travel-Recovery Upgrade", "Better sleep at home."),
    ("Marisol Ortega", 35, "Mom", "Mid-Life Mom Upgrade", "Treating self."),
    ("Linnea Eriksson", 38, "RN", "Healthcare-Pro Upgrade", "Knows sleep matters."),
    ("Beau Marchetti", 32, "Architect", "Design-Forward Buyer", "Aesthetic + function."),
    ("Otis Reynolds", 50, "Pre-Retiree", "Retirement Setup", "Last-mattress purchase."),
    ("Ines Ribeiro", 38, "Mom of Twins", "Family-Need Upgrade", "Sleeping well finally."),
]

MATTRESS_BACK_PAIN: list[Persona] = [
    ("Carlos Mendoza", 40, "Manager", "Chronic-Back Pain", "Doctor-recommended firm."),
    ("Reginald Cross", 45, "Manager", "Lower-Back Sufferer", "Years of issues."),
    ("Naomi Ferguson", 40, "Athlete", "Athletic-Injury Recovery", "Old injury management."),
    ("Devon Sutherland", 42, "Sales", "Standing-Job Sufferer", "Long days on feet."),
    ("Elena Castellanos", 35, "Teacher", "Standing-Day Pain", "Classroom on feet."),
    ("Hiroshi Tanaka", 35, "Gym Owner", "Athletic-Stress Pain", "Heavy training history."),
    ("Diego Vargas", 28, "Powerlifter", "Lift-Recovery Buyer", "Heavy back squat history."),
    ("Felix Aurelius", 33, "Olympic Lifter", "Snatch-Recovery Buyer", "Spinal-load athlete."),
    ("Otis Reynolds", 50, "Pre-Retiree", "Aging-Back Buyer", "Decades of office work."),
    ("Linnea Eriksson", 38, "RN", "Nursing-Pain Pro", "Patient lifting damage."),
    ("Marisol Ortega", 38, "Mom", "Postpartum-Back Buyer", "After-kids back pain."),
    ("Camille Faure", 30, "Acupuncturist", "Practitioner Pain Pro", "Treats own + clients."),
    ("Owen Sinclair", 30, "Architect", "Desk-Posture Sufferer", "Long screen days."),
    ("Trevor Bishop", 35, "IT Manager", "Sitting-All-Day Pain", "Office worker."),
    ("Jonas Albrecht", 42, "Executive", "Stress-Posture Pain", "C-suite chronic."),
]


# ═══════════════════════════════════════════════════════════════════════
# PREMIUM DRINKWARE (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

DRINKWARE_OUTDOOR: list[Persona] = [
    ("Lucas Nakamura", 32, "Hiker", "Multi-Day Backpacker", "Insulated bottle for trail."),
    ("Yara Khalil", 28, "Climber", "Outdoor Climber", "Multi-pitch days."),
    ("Naomi Ferguson", 40, "Triathlete", "Endurance Outdoor", "Long-format athlete."),
    ("Marcus Johnson", 32, "Trainer", "Outdoor Trainer", "Outdoor bootcamps."),
    ("Diego Vargas", 28, "Powerlifter", "Garage Gym Athlete", "Shed-gym essentials."),
    ("Beau Marchetti", 32, "CrossFit", "Outdoor WOD-er", "Box outdoor sessions."),
    ("Saanvi Krishnan", 26, "Bodybuilder", "Outdoor Cardio Athlete", "Park sprints."),
    ("Devon Sutherland", 42, "Sales", "Outdoor-Travel Sales", "Site visits + trail."),
    ("Hiroshi Tanaka", 35, "Owner", "Outdoor-Class Owner", "Park bootcamps."),
    ("Owen Kelleher", 25, "Rugby Player", "Field-Sport Athlete", "Pitch hydration."),
    ("Adaeze Okafor", 27, "Coach", "Field-Coach Pro", "Sideline supplies."),
    ("Felix Aurelius", 33, "Olympic Lifter", "Park-Lift Pro", "Mobile gym setup."),
    ("Talia Rosenberg", 28, "Hybrid Athlete", "Outdoor-Yoga Pro", "Park yoga teacher."),
    ("Xavier Beaumont", 23, "Track Athlete", "Track Field Athlete", "Outdoor track sessions."),
    ("Ravi Subramanian", 30, "Ultra Runner", "Trail-Race Pro", "Mountain races."),
]

DRINKWARE_OFFICE: list[Persona] = [
    ("Mei Lin Zhao", 28, "Marketing Manager", "Office Hydration Pro", "Desk water bottle."),
    ("Theo Whitaker", 38, "Tech Lead", "WFH-Office Hybrid", "Desk + commute."),
    ("Maya Goldberg", 27, "Designer", "Aesthetic-First Buyer", "Pretty desk drinkware."),
    ("Priya Patel", 30, "PM", "Daily-Use Buyer", "8-hour office bottle."),
    ("Adaeze Okafor", 27, "PR", "Travel-Office Pro", "Plane-friendly bottle."),
    ("Brigitte Laurent", 38, "Director", "Premium-Office Buyer", "Top-tier brand."),
    ("Niamh Doyle", 29, "Pharmacist", "Healthcare-Office Buyer", "All-day shift bottle."),
    ("Felix Aurelius", 33, "Engineer", "Engineer-Mindset Buyer", "Insulation-focus."),
    ("Trevor Bishop", 35, "IT Manager", "Practical Office Buyer", "Indestructible required."),
    ("Owen Sinclair", 30, "Architect", "Design-Forward Office", "Object-as-art."),
    ("Talia Rosenberg", 28, "Therapist", "Office-Wellness Buyer", "Hydration in mental health practice."),
    ("Saanvi Krishnan", 26, "Founder", "Hustle-Culture Buyer", "Founder-status drinkware."),
    ("Linnea Eriksson", 38, "RN", "Hospital-Office Pro", "12-hour shift bottle."),
    ("Camille Dupree", 32, "Yoga Manager", "Studio + Office Hybrid", "Class to desk."),
    ("Reginald Cross", 45, "Manager", "Senior-Pro Buyer", "Premium tier comfort."),
]

DRINKWARE_FITNESS: list[Persona] = [
    ("Brooke Stephens", 30, "Trainer", "Gym-Bag Essential", "Daily gym bottle."),
    ("Marcus Johnson", 32, "CrossFit Coach", "Box Bottle", "Affiliate of choice."),
    ("Diego Vargas", 28, "Powerlifter", "Garage Lifter", "Pre-workout bottle."),
    ("Saanvi Krishnan", 26, "Bodybuilder", "Show-Prep Athlete", "Always carrying."),
    ("Hiroshi Tanaka", 35, "Gym Owner", "Owner-Branded Drinkware", "Gym branded."),
    ("Felix Aurelius", 33, "Olympic Lifter", "Plate-Loaded Athlete", "Heavy-day hydration."),
    ("Beau Marchetti", 32, "Functional Athlete", "Hyrox Bottle", "Long-format competitor."),
    ("Adaeze Okafor", 27, "Coach", "Pro-Coach Drinkware", "Multiple bottles tracked."),
    ("Owen Kelleher", 25, "Rugby Player", "Pitch Bottle", "Match-day essential."),
    ("Talia Rosenberg", 28, "Hybrid Athlete", "Multi-Sport Bottle", "Pilates + lifting."),
    ("Lucas Nakamura", 32, "BJJ", "Mat-Bag Bottle", "Roll-day bottle."),
    ("Xavier Beaumont", 23, "Track Athlete", "NCAA Bottle", "Team-issued + personal."),
    ("Yara Khalil", 28, "Climber", "Crag Bottle", "Approach + sport day."),
    ("Naomi Ferguson", 40, "Master's Triathlete", "Master's Bottle", "Multi-sport carrier."),
    ("Ravi Subramanian", 30, "Ultra Runner", "Race-Day Bottle", "Aid-station refill."),
]


# ═══════════════════════════════════════════════════════════════════════
# PREMIUM BASICS / APPAREL (3 sub-pools × 15 = 45)
# ═══════════════════════════════════════════════════════════════════════

APPAREL_BASICS_MINIMALISTS: list[Persona] = [
    ("Owen Sinclair", 30, "Designer", "Capsule-Wardrobe Pro", "Ten-piece wardrobe."),
    ("Beau Marchetti", 32, "Architect", "Quiet-Luxury Buyer", "Investment basics."),
    ("Maya Goldberg", 27, "Designer", "Aesthetic-Minimalist", "Clean lines only."),
    ("Theo Whitaker", 38, "Tech Lead", "Tech-Bro Capsule", "Same outfit daily."),
    ("Felix Aurelius", 33, "Engineer", "Decision-Fatigue Buyer", "Removes choices."),
    ("Brigitte Laurent", 38, "Director", "Premium-Minimalist", "Quality + simplicity."),
    ("Yara Khalil", 28, "Stylist", "Pro-Minimalist Stylist", "Educates clients."),
    ("Niamh Doyle", 29, "Pharmacist", "Functional Minimalist", "Work + life basics."),
    ("Soren Bjorklund", 30, "Engineer", "Scandinavian Minimalist", "Cultural minimalist."),
    ("Lucas Nakamura", 32, "Tech", "Tech-Pro Minimalist", "Uniform-mindset."),
    ("Adaeze Okafor", 27, "PR", "Industry-Pro Minimalist", "Polished minimalism."),
    ("Camille Dupree", 32, "Yoga Manager", "Studio-Lifestyle Minimalist", "Yoga + street basics."),
    ("Priya Patel", 30, "PM", "Travel-Pro Minimalist", "Carry-on wardrobe."),
    ("Talia Rosenberg", 28, "Therapist", "Practice-Wear Minimalist", "Office + life same."),
    ("Hiroshi Tanaka", 35, "Owner", "Owner-Uniform Buyer", "Same outfit running gym."),
]

APPAREL_BASICS_QUALITY_SEEKERS: list[Persona] = [
    ("Reginald Cross", 45, "Manager", "Investment-Buyer", "Buy once, last decade."),
    ("Robert Chen", 40, "Executive", "Premium-Quality Buyer", "Top-tier basics."),
    ("Brigitte Laurent", 38, "Director", "Made-Well Buyer", "Origin + quality matter."),
    ("Owen Sinclair", 30, "Architect", "Heritage-Brand Buyer", "Storied makers only."),
    ("Beau Marchetti", 32, "Architect", "Construction-Forward Buyer", "Stitching matters."),
    ("Naomi Ferguson", 40, "VP", "Career-Wardrobe Buyer", "Investing in self."),
    ("Theo Whitaker", 38, "Tech Lead", "Quiet-Luxury Quality", "Brand-agnostic quality."),
    ("Jonas Albrecht", 42, "C-Suite", "Premium-Tier Quality", "Top-shelf basics."),
    ("Maya Goldberg", 27, "Designer", "Aesthetic-Quality Buyer", "Design + construction."),
    ("Yara Khalil", 28, "Stylist", "Pro-Stylist Quality", "Recommends quality clients."),
    ("Lucas Nakamura", 32, "Tech", "Career-Pro Quality", "Investment basics."),
    ("Priya Patel", 30, "PM", "Career-Wardrobe Pro", "Buying for the role."),
    ("Brigitte Faure", 35, "Architect", "Heritage-Maker Buyer", "Decades-old brands."),
    ("Adaeze Okafor", 27, "PR", "Industry-Pro Quality", "Polished + lasting."),
    ("Devon Sutherland", 42, "Sales", "Travel-Pro Quality", "Wear-for-years pieces."),
]

APPAREL_BASICS_SUSTAINABLE: list[Persona] = [
    ("Linnea Eriksson", 38, "RN", "Eco-Conscious Pro", "Reads supply chain."),
    ("Brigitte Laurent", 38, "Coach", "Sustainability-Educator", "Educates clients."),
    ("Maya Goldberg", 27, "Designer", "Eco-Designer Buyer", "Sustainable-design buyer."),
    ("Yara Khalil", 28, "Stylist", "Slow-Fashion Stylist", "Clients toward sustainable."),
    ("Priya Patel", 30, "PM", "Climate-Conscious Pro", "Carbon-aware shopper."),
    ("Niamh Doyle", 29, "Pharmacist", "Eco-Pharmacist", "Health + planet."),
    ("Talia Rosenberg", 28, "Therapist", "Mindful-Consumer Therapist", "Practices mindfulness."),
    ("Camille Dupree", 32, "Yoga Manager", "Eco-Yoga Studio", "Studio-values aligned."),
    ("Adaeze Okafor", 27, "PR", "Industry-Sustainability Pro", "Knows green-washing."),
    ("Owen Sinclair", 30, "Architect", "Sustainable-Design Pro", "LEED-mindset."),
    ("Soren Bjorklund", 30, "Engineer", "Scandinavian-Eco Pro", "Cultural sustainability."),
    ("Eun-ji Park", 25, "Designer", "Eco-K-Design Buyer", "Korean sustainable design."),
    ("Saanvi Krishnan", 26, "Founder", "Mission-Aligned Buyer", "Founder values aligned."),
    ("Camille Faure", 30, "Acupuncturist", "Holistic-Eco Practitioner", "Whole-system thinking."),
    ("Brigitte Faure", 35, "Architect", "Eco-Architect Buyer", "Profession-aligned."),
]


# ═══════════════════════════════════════════════════════════════════════
# SCHOOL SUPPLIES (5 sub-pools × 12 = 60)
# ═══════════════════════════════════════════════════════════════════════

SCHOOL_STUDENTS: list[Persona] = [
    ("Ava Mitchell", 17, "High School Junior", "AP-Student Notebook Buyer", "Color-codes by subject."),
    ("Ethan Park", 16, "High School Sophomore", "Math-Heavy Student", "Graph paper essential."),
    ("Sophia Williams", 17, "Senior", "College-Bound Student", "Bullet-journal user."),
    ("Marcus Lee", 15, "Freshman", "Intro-Student Buyer", "First-time own buyer."),
    ("Aisha Robinson", 16, "Sophomore", "Honors Student", "Highlighter + notebook system."),
    ("Noah Garcia", 17, "Junior", "STEM Student", "Specific notebook needs."),
    ("Olivia Chen", 16, "Sophomore", "Art-Class Student", "Sketchbook + class notebooks."),
    ("Lucas Brown", 15, "Freshman", "First-Year Buyer", "Mom-helped picker."),
    ("Maya Singh", 17, "Senior", "AP-Heavy Student", "5 different notebooks."),
    ("Jamie Davis", 14, "Eighth Grader", "Middle-School Buyer", "Transitioning to high school."),
    ("Chloe Kim", 16, "Sophomore", "Aesthetic-First Student", "Picks for looks."),
    ("Tyler Anderson", 17, "Junior", "Sports + School Student", "Multi-purpose notebook."),
]

SCHOOL_PARENTS: list[Persona] = [
    ("Carlos Mendoza", 42, "Father of 3", "Multi-Kid Supply Buyer", "August list shopper."),
    ("Elena Castellanos", 38, "Working Mom", "Annual School Buyer", "Bulk-buy season."),
    ("Marisol Ortega", 35, "Single Mom", "Budget-Conscious Buyer", "Off-brand acceptable."),
    ("Naomi Ferguson", 42, "Mom of Teens", "High-School-Era Buyer", "Specific subject needs."),
    ("Reginald Cross", 47, "Dad", "Empty-Nest-Adjacent Dad", "Last-kid era."),
    ("Isabella Santos", 34, "Mom of 2", "Twin-Buyer Mom", "Doubles everything."),
    ("Trevor Bishop", 38, "Stay-at-Home Dad", "Primary-Caregiver Dad", "Lists are sacred."),
    ("Ines Ribeiro", 40, "Mom of 4", "Bulk-Family Buyer", "Costco runs."),
    ("Jonas Albrecht", 45, "Executive Dad", "Premium-Supply Buyer", "Top-tier preferences."),
    ("Devon Sutherland", 44, "Coach Dad", "Sports + School Multitasker", "Both school and team gear."),
    ("Camille Dupree", 35, "Yoga Manager Mom", "Wellness-Aligned Mom", "Eco supplies preferred."),
    ("Priya Patel", 32, "Tech Mom", "Career-Mom Buyer", "Online ordering."),
]

SCHOOL_TEACHERS: list[Persona] = [
    ("Elena Castellanos", 38, "5th Grade Teacher", "Elementary Teacher", "Buys for own classroom."),
    ("Talia Rosenberg", 30, "High School English", "Literature Teacher", "Composition books."),
    ("Marcus Pierce", 35, "Middle School Math", "STEM Teacher", "Graph + composition."),
    ("Linnea Eriksson", 38, "Special Ed", "Special-Ed Teacher", "Adaptive materials."),
    ("Devon Sutherland", 42, "PE Teacher", "Coach-Teacher", "Multi-purpose buyer."),
    ("Brigitte Laurent", 40, "Spanish Teacher", "Language Teacher", "Vocab notebooks."),
    ("Owen Sinclair", 32, "Art Teacher", "Studio Teacher", "Sketchbook + class."),
    ("Naomi Ferguson", 42, "AP History", "AP-Course Teacher", "Specific test prep."),
    ("Reginald Cross", 47, "Department Chair", "Veteran Teacher", "30-year buyer."),
    ("Felix Aurelius", 33, "Physics Teacher", "Lab-Notebook Buyer", "Science-specific."),
    ("Maya Goldberg", 27, "First-Year Teacher", "New-Teacher Buyer", "Setting up classroom."),
    ("Camille Faure", 30, "Yoga + School Teacher", "Hybrid Educator", "Multiple settings."),
]

SCHOOL_COLLEGE: list[Persona] = [
    ("Tyler Brennan", 22, "Senior", "Final-Year Buyer", "Last college supply run."),
    ("Connor O'Brien", 20, "Junior", "STEM Major", "Engineering notebook needs."),
    ("Diana Velasquez", 21, "Pre-Med Junior", "Med-Track Buyer", "MCAT-prep notes."),
    ("Eun-ji Park", 22, "Senior", "Studio-Major Buyer", "Sketchbook-heavy major."),
    ("Maya Patel", 20, "Sophomore", "Liberal-Arts Major", "Composition-heavy."),
    ("Jamie Reyes", 19, "Sophomore", "First-Year-Adjacent Buyer", "Settled into routine."),
    ("Trevor Lin", 23, "Senior", "Pre-Med Senior", "Final-stretch buyer."),
    ("Sophia Castro", 19, "Freshman", "First-Year College Buyer", "Just learning."),
    ("Asha Mehta", 24, "Master's Student", "Grad-Student Supplier", "Different-tier needs."),
    ("Owen Kelleher", 21, "Junior", "Athlete-Student Buyer", "Sport + study."),
    ("Talia Bergman", 20, "Sophomore", "Psychology Major", "Lecture-heavy major."),
    ("Camille Faure", 25, "PhD First-Year", "Doc-Track Buyer", "Research-grade buyer."),
]

SCHOOL_AESTHETIC: list[Persona] = [
    ("Sophia Russo", 24, "Designer", "Bullet-Journal Buyer", "Aesthetic-first."),
    ("Maya Goldberg", 27, "Designer", "Pretty-Notebook Buyer", "Sees notebooks as decor."),
    ("Eun-ji Park", 22, "Studio Major", "K-Stationery Fan", "Imports Korean."),
    ("Yara Khalil", 28, "Stylist", "Pro-Aesthetic Buyer", "Stationery as identity."),
    ("Lucia Romano", 26, "Content Creator", "Aesthetic-Content Buyer", "Notebook visible on stream."),
    ("Mei-ling Wu", 23, "Designer", "Visual-Identity Buyer", "Notebooks for shoots."),
    ("Camille Dupree", 28, "Yoga Manager", "Wellness-Aesthetic Buyer", "Studio-lifestyle aligned."),
    ("Camille Faure", 30, "Acupuncturist", "TCM-Aesthetic Buyer", "Patient-notes aesthetic."),
    ("Adaeze Okafor", 27, "PR", "Industry-Aesthetic Pro", "Polished journals."),
    ("Brigitte Laurent", 38, "Coach", "Wellness-Journal Buyer", "Practice journals."),
    ("Talia Rosenberg", 28, "Therapist", "Therapy-Journal Buyer", "Patient-facing notebooks."),
    ("Niamh Doyle", 29, "Pharmacist", "Functional-Aesthetic", "Pretty + functional."),
]


# ═══════════════════════════════════════════════════════════════════════
# RELIGIOUS LIFESTYLE (3 sub-pools × 12 = 36)
# ═══════════════════════════════════════════════════════════════════════

RELIGIOUS_LIFESTYLE_MUSLIM_PROFESSIONALS: list[Persona] = [
    ("Fatima Hassan", 28, "Nurse", "Practicing-Muslim Pro", "Five prayers around shifts."),
    ("Yara Khalil", 28, "Marketing Pro", "Professional Muslimah", "Office prayer challenges."),
    ("Aisha Hassan", 35, "Architect", "Mid-Career Muslimah", "Established prayer routine."),
    ("Nadia Khouri", 30, "Public Health Pro", "Field-Worker Muslim", "Travel + prayer."),
    ("Adaeze Mohammed", 29, "PR Director", "Industry-Pro Muslim", "Industry prayer challenges."),
    ("Asha Mehta", 28, "Pharmacy Pro", "Healthcare Muslim", "Hospital prayer logistics."),
    ("Saanvi Krishnan", 26, "Founder", "Muslim Founder", "Startup + faith."),
    ("Maya Patel", 25, "Marketing", "Junior-Pro Muslim", "Office prayer accommodation."),
    ("Priya Patel", 30, "Researcher", "Academic Muslim", "Conference-travel prayer."),
    ("Camille Faure", 30, "Acupuncturist", "Healthcare Muslim", "Patient-care + prayer."),
    ("Trevor Lin", 26, "Med Resident", "Med-Resident Muslim", "Long shifts + prayer."),
    ("Camille Dupree", 28, "Yoga Manager", "Wellness Muslim", "Faith + studio life."),
]

RELIGIOUS_LIFESTYLE_MUSLIM_FAMILIES: list[Persona] = [
    ("Carlos Mohammed", 42, "Father of 3", "Family-Patriarch Muslim", "Teaching kids prayer."),
    ("Marisol Hassan", 38, "Mom of 2", "Practicing-Mom Muslim", "Family prayer routine."),
    ("Reginald Khalid", 47, "Empty-Nest Dad", "Mature Muslim Dad", "Established practice."),
    ("Isabella Mohammed", 34, "New Mom", "Postpartum Muslim", "Adjusting prayer with baby."),
    ("Asha Mehta-Hassan", 28, "Pregnant Mom", "Expecting Muslim", "Pregnancy prayer adaptation."),
    ("Naomi Khalid", 42, "Mom of Teens", "Teen-Parent Muslim", "Teaching teens responsibility."),
    ("Trevor Bishop-Hassan", 38, "Convert Dad", "New-Convert Family", "Learning + teaching."),
    ("Ines Hassan", 40, "Mom of 4", "Multi-Kid Family", "Bulk-buyer Muslim."),
    ("Jonas Mohammed", 45, "Executive Dad", "Pro-Father Muslim", "Career + faith balance."),
    ("Devon Sutherland-Hassan", 44, "Coach Dad", "Coach-Father Muslim", "Sports + faith."),
    ("Camille Mohammed", 35, "Yoga Mom", "Wellness Muslim Mom", "Studio + family + faith."),
    ("Priya Mohammed", 32, "Tech Mom", "Career Muslim Mom", "Remote work + prayer."),
]

RELIGIOUS_LIFESTYLE_SMART_HOME: list[Persona] = [
    ("Felix Aurelius", 33, "Engineer", "Smart-Home Enthusiast", "Apple HomeKit power-user."),
    ("Hiroshi Tanaka", 35, "Owner", "Tech-Forward Muslim", "Bridges tech + tradition."),
    ("Soren Bjorklund", 30, "Bioinformatics", "Data-Smart-Home Pro", "Quantified household."),
    ("Theo Whitaker", 38, "Tech Lead", "Premium Smart-Home", "Top-tier devices."),
    ("Diego Vargas", 28, "Software Engineer", "Code-Heavy Smart-Home", "Custom integrations."),
    ("Lucas Nakamura", 32, "Tech Pro", "Tech-Pro Smart-Home", "Multiple ecosystems."),
    ("Saanvi Krishnan", 26, "Founder", "Founder Smart-Home", "Office + home automation."),
    ("Beau Marchetti", 32, "Architect", "Design-Forward Smart-Home", "Aesthetic integrations."),
    ("Owen Sinclair", 30, "Architect", "Pro-Architect Smart-Home", "Spec'ing for clients."),
    ("Ravi Subramanian", 30, "Quant", "Data-Driven Smart-Home", "Spreadsheets the home."),
    ("Adaeze Okafor", 27, "PR", "Industry-Aware Smart-Home", "Knows the brands."),
    ("Trevor Bishop", 35, "IT Manager", "IT-Pro Smart-Home", "Network-first thinker."),
]


# ═══════════════════════════════════════════════════════════════════════
# PET PRODUCTS (3 sub-pools × 12 = 36)
# ═══════════════════════════════════════════════════════════════════════

PET_DOG_OWNERS: list[Persona] = [
    ("Brooke Stephens", 30, "Trainer", "Active-Dog Owner", "Athletic-dog routine."),
    ("Marcus Johnson", 32, "CrossFit Coach", "High-Energy Dog Owner", "Working-breed dog."),
    ("Elena Castellanos", 35, "Teacher", "Family-Dog Owner", "Family lab."),
    ("Carlos Mendoza", 40, "Manager", "Multi-Dog Family", "Two dogs to feed."),
    ("Hiroshi Tanaka", 35, "Gym Owner", "Owner-Dog Team", "Dog at gym daily."),
    ("Naomi Ferguson", 40, "VP", "Senior-Dog Owner", "Aging-dog needs."),
    ("Felix Aurelius", 33, "Engineer", "Engineer-Pet Owner", "Optimization mindset."),
    ("Talia Rosenberg", 28, "Therapist", "Therapy-Dog Owner", "Working dog with role."),
    ("Owen Sinclair", 30, "Architect", "Apartment-Dog Owner", "City-dog logistics."),
    ("Brigitte Laurent", 38, "Coach", "Wellness-Dog Owner", "Premium-tier dog."),
    ("Trevor Bishop", 35, "IT Manager", "Suburban-Dog Dad", "Family lab routine."),
    ("Yara Khalil", 28, "Stylist", "Aesthetic-Dog Owner", "Curated dog life."),
]

PET_CAT_OWNERS: list[Persona] = [
    ("Maya Goldberg", 27, "Designer", "Aesthetic-Cat Owner", "Cat-content creator."),
    ("Camille Dupree", 32, "Yoga Manager", "Studio-Lifestyle Cat", "Apartment cat."),
    ("Sophia Russo", 28, "Beauty Editor", "Pro-Cat Owner", "Two-cat household."),
    ("Eun-ji Park", 25, "Designer", "K-Cat Lifestyle", "Cat-as-decor culture."),
    ("Owen Sinclair", 30, "Architect", "Pro-Cat Architect", "Apartment-only owner."),
    ("Adaeze Okafor", 27, "PR", "Industry-Pro Cat Owner", "Solo professional."),
    ("Lucas Nakamura", 32, "Tech", "Tech-Pro Cat Owner", "WFH + cats."),
    ("Saanvi Krishnan", 26, "Founder", "Founder Cat-Mom", "Startup + cats."),
    ("Soren Bjorklund", 30, "Engineer", "Engineer-Cat Owner", "Optimization mindset."),
    ("Camille Faure", 30, "Acupuncturist", "Holistic Cat Owner", "Whole-cat wellness."),
    ("Talia Rosenberg", 28, "Therapist", "Apartment-Cat Owner", "City-cat routine."),
    ("Lucia Romano", 26, "Content Creator", "Cat-Content Creator", "Cats on stream."),
]

PET_PREMIUM: list[Persona] = [
    ("Brigitte Laurent", 38, "Director", "Premium-Pet Buyer", "Top-tier everything."),
    ("Robert Chen", 40, "Executive", "Investment-Pet Buyer", "Premium-only."),
    ("Naomi Ferguson", 40, "VP", "Senior-Pet Premium", "Aging-pet investment."),
    ("Reginald Cross", 47, "Manager", "Mature-Pet Pro", "Premium-tier loyal."),
    ("Maya Goldberg", 27, "Designer", "Aesthetic-Premium Pro", "Visual + quality."),
    ("Yara Khalil", 28, "Stylist", "Lifestyle-Premium Pro", "Pet as accessory."),
    ("Brigitte Faure", 35, "Architect", "Heritage-Brand Pet", "Decades-old brands."),
    ("Owen Sinclair", 30, "Architect", "Design-Forward Pet", "Object-as-art pet."),
    ("Adaeze Okafor", 27, "PR", "Industry-Pro Premium", "Knows pet brands."),
    ("Hiroshi Tanaka", 35, "Owner", "Operator-Premium Pet", "Premium owner."),
    ("Theo Whitaker", 38, "Tech Lead", "Tech-Premium Pet", "Quantified pet."),
    ("Linnea Eriksson", 38, "RN", "Veterinary-Aware Premium", "Health-first premium."),
]


# ═══════════════════════════════════════════════════════════════════════
# BABY / FAMILY (3 sub-pools × 12 = 36)
# ═══════════════════════════════════════════════════════════════════════

BABY_FAMILY_NEW_PARENTS: list[Persona] = [
    ("Asha Mehta", 28, "Pregnant Pro", "Expecting Pro", "Researching obsessively."),
    ("Soren Bjorklund", 30, "Engineer", "First-Time Dad", "Spreadsheets parenting."),
    ("Yara Khalil", 28, "First-Time Mom", "New Mom", "Postpartum buyer."),
    ("Saanvi Krishnan", 26, "Founder", "Pregnant Founder", "Startup + pregnancy."),
    ("Camille Faure", 30, "Acupuncturist", "Health-Pro Pregnant", "Holistic pregnancy."),
    ("Trevor Bishop", 35, "IT Manager", "Late-Dad", "First child at 35."),
    ("Maya Patel", 25, "Marketing Junior", "Young-Mom", "First-pregnancy buyer."),
    ("Niamh Doyle", 29, "Pharmacist", "Pharmacist-Mom", "Medication-aware buyer."),
    ("Adaeze Okafor", 27, "PR", "Industry-Pregnant Pro", "Career + pregnancy."),
    ("Diana Velasquez", 27, "Doctor", "Doctor-Mom", "Medical-perspective buyer."),
    ("Camille Dupree", 32, "Yoga Manager", "Wellness Pregnant", "Studio-lifestyle pregnancy."),
    ("Brigitte Laurent", 38, "Director", "Late-Bloomer Mom", "First child at 38."),
]

BABY_FAMILY_EXPERIENCED: list[Persona] = [
    ("Carlos Mendoza", 42, "Father of 3", "Multi-Kid Veteran", "Bulk-buyer."),
    ("Elena Castellanos", 38, "Working Mom", "Annual-Cycle Buyer", "Knows what works."),
    ("Marisol Ortega", 35, "Single Mom", "Solo-Parent Veteran", "Budget veteran."),
    ("Naomi Ferguson", 42, "Mom of Teens", "Past-Baby-Years Mom", "Looking back-ish."),
    ("Reginald Cross", 47, "Empty-Nest Dad", "Pet-Phase Empty-Nester", "Past parenting."),
    ("Isabella Santos", 34, "Mom of 2", "Mom-of-2 Pro", "Hand-me-down system."),
    ("Trevor Bishop", 38, "Stay-at-Home Dad", "Primary-Caregiver Veteran", "Daily-routine pro."),
    ("Ines Ribeiro", 40, "Mom of Twins", "Twin-Mom Veteran", "Bulk-everything."),
    ("Devon Sutherland", 44, "Coach Dad", "Active-Dad Veteran", "Sports-family veteran."),
    ("Camille Dupree", 35, "Yoga Mom", "Wellness Mom Pro", "Studio + family veteran."),
    ("Priya Patel", 32, "Tech Mom", "Career-Mom Pro", "Logistics-pro."),
    ("Linnea Eriksson", 40, "RN Mom", "Healthcare Mom Pro", "Medical-aware veteran."),
]

BABY_FAMILY_ORGANIC: list[Persona] = [
    ("Brigitte Laurent", 38, "Coach", "Holistic Mom", "Organic-only."),
    ("Maya Goldberg", 27, "Designer", "Eco-Mom", "Sustainable parenting."),
    ("Camille Dupree", 32, "Yoga Manager", "Wellness Mom", "Whole-foods baby."),
    ("Talia Rosenberg", 28, "Therapist", "Mindful Parent", "Conscious choices."),
    ("Saanvi Krishnan", 26, "Founder", "Founder Mom", "Premium-organic."),
    ("Niamh Doyle", 29, "Pharmacist", "Health-First Mom", "Evidence + organic."),
    ("Linnea Eriksson", 38, "RN", "Healthcare Mom", "Medical + organic."),
    ("Camille Faure", 30, "Acupuncturist", "Holistic Practitioner Mom", "TCM + organic."),
    ("Yara Khalil", 28, "Stylist", "Aesthetic-Eco Mom", "Pretty organic."),
    ("Brigitte Faure", 35, "Architect", "Eco-Architect Mom", "Sustainable household."),
    ("Adaeze Okafor", 27, "PR", "Industry-Eco Pro Mom", "Knows green-washing."),
    ("Naomi Ferguson", 42, "VP", "Eco-Conscious Mom of Teens", "Long-term sustainability."),
]


# ═══════════════════════════════════════════════════════════════════════
# GENERIC FALLBACK BANKS
# ═══════════════════════════════════════════════════════════════════════

FOOD_BEVERAGE_GENERIC: list[Persona] = [
    ("Mei Lin Zhao", 28, "Marketing Manager", "Wellness-Conscious Pro", "Health-driven shopper."),
    ("Carlos Mendoza", 40, "Manager", "Family Buyer", "Whole-family preferences."),
    ("Maya Goldberg", 27, "Designer", "Trend-Aware Buyer", "Tries new launches."),
    ("Reginald Cross", 47, "Manager", "Mature Adult Buyer", "Steady preferences."),
    ("Brooke Stephens", 30, "Trainer", "Active Adult", "Performance-focused."),
    ("Brigitte Laurent", 38, "Director", "Premium Health Buyer", "Top-shelf only."),
    ("Tyler Brennan", 22, "Recent Grad", "Young Adult Buyer", "Budget + quality."),
    ("Felix Aurelius", 33, "Engineer", "Optimization Buyer", "Data-driven."),
    ("Naomi Ferguson", 40, "VP", "Career-Mom Buyer", "Family + work."),
    ("Theo Whitaker", 38, "Tech Lead", "Performance Pro", "Cognition-focused."),
    ("Camille Dupree", 32, "Yoga Manager", "Studio-Lifestyle Buyer", "Wellness aligned."),
    ("Hiroshi Tanaka", 35, "Owner", "Operator-Athlete", "Test-everything."),
    ("Adaeze Okafor", 27, "PR", "Industry-Aware Pro", "Knows brands."),
    ("Diego Vargas", 28, "Lawyer", "Career-Focused Pro", "Convenience-first."),
    ("Talia Rosenberg", 28, "Therapist", "Wellness-Practice Pro", "Health-first."),
]

BEAUTY_PERSONAL_GENERIC: list[Persona] = [
    ("Sophia Russo", 28, "Beauty Editor", "Industry Pro", "Tries new launches."),
    ("Maya Goldberg", 27, "Designer", "Trend-Aware Buyer", "Routine-evolver."),
    ("Priya Patel", 30, "PA", "Pro-Adjacent Buyer", "Knows the science."),
    ("Camille Dupree", 32, "Yoga Manager", "Wellness Buyer", "Holistic aligned."),
    ("Talia Rosenberg", 28, "Makeup Artist", "Pro Buyer", "Tests on shoots."),
    ("Eun-ji Park", 25, "Designer", "K-Beauty Buyer", "Korean focus."),
    ("Brigitte Laurent", 38, "Coach", "Premium Buyer", "Top-tier loyal."),
    ("Yara Khalil", 28, "Stylist", "Pro-Stylist Buyer", "Recommends to clients."),
    ("Adaeze Okafor", 27, "PR", "Industry Insider", "PR samples."),
    ("Saanvi Krishnan", 26, "Founder", "Indie-Brand Tester", "Knows formulators."),
    ("Niamh Doyle", 29, "Pharmacist", "Active-Aware Buyer", "Verifies claims."),
    ("Camille Faure", 30, "Esthetician", "Spa Owner", "Educated retail."),
    ("Lucia Romano", 26, "Vlogger", "Reviewer", "Long-form analysis."),
    ("Mei-ling Wu", 23, "Designer", "Visual-Identity Buyer", "Aesthetic-first."),
    ("Asha Mehta", 28, "Beauty Tech", "Indie Founder", "Reformulates."),
]

HOME_LIFESTYLE_GENERIC: list[Persona] = [
    ("Owen Sinclair", 30, "Architect", "Design-Forward Buyer", "Object-as-art."),
    ("Beau Marchetti", 32, "Architect", "Quiet-Luxury Buyer", "Investment pieces."),
    ("Maya Goldberg", 27, "Designer", "Aesthetic Buyer", "Clean lines."),
    ("Theo Whitaker", 38, "Tech Lead", "Premium Buyer", "Top-tier."),
    ("Brigitte Laurent", 38, "Director", "Quality-First Buyer", "Buy once well."),
    ("Carlos Mendoza", 40, "Manager", "Family-Home Buyer", "Whole-family use."),
    ("Hiroshi Tanaka", 35, "Owner", "Operator Buyer", "Tests for biz."),
    ("Reginald Cross", 47, "Manager", "Mature Home Buyer", "Established preferences."),
    ("Yara Khalil", 28, "Stylist", "Pro-Stylist Buyer", "Educates clients."),
    ("Felix Aurelius", 33, "Engineer", "Functional-First Buyer", "Engineering mindset."),
    ("Brigitte Faure", 35, "Architect", "Heritage Buyer", "Decades-old brands."),
    ("Lucas Nakamura", 32, "Tech", "Career-Pro Buyer", "Pro home setup."),
    ("Adaeze Okafor", 27, "PR", "Industry Pro", "Polished spaces."),
    ("Camille Dupree", 32, "Yoga Manager", "Wellness-Aligned Buyer", "Studio aesthetic."),
    ("Robert Chen", 40, "Executive", "Investment Buyer", "Premium tier."),
]

TECH_WELLNESS_GENERIC: list[Persona] = [
    ("Felix Aurelius", 33, "Engineer", "Quantified-Self Buyer", "Wearable + tracker."),
    ("Hiroshi Tanaka", 35, "Owner", "Athletic-Tech Buyer", "Performance-track."),
    ("Theo Whitaker", 38, "Tech Lead", "Premium-Tech Buyer", "Top-tier wearables."),
    ("Soren Bjorklund", 30, "Bioinformatics", "Data-Heavy Buyer", "Tracks everything."),
    ("Saanvi Krishnan", 26, "Founder", "Founder-Optimizer", "Career-optimization tech."),
    ("Lucas Nakamura", 32, "Tech Pro", "Tech-Career Buyer", "Multi-device."),
    ("Diego Vargas", 28, "Software Engineer", "Engineer-Tech Buyer", "Custom integrations."),
    ("Brooke Stephens", 30, "Trainer", "Athletic-Tech Buyer", "Performance tracking."),
    ("Maya Goldberg", 27, "Designer", "Aesthetic-Tech Buyer", "Design-aware tech."),
    ("Akira Sato", 36, "Senior Engineer", "Cognition-Tech Buyer", "Brain-focus tech."),
    ("Marcus Johnson", 32, "Coach", "Coach-Tech Buyer", "Athlete + coach data."),
    ("Adaeze Okafor", 27, "PR", "Industry-Aware Tech", "Knows brands."),
    ("Owen Sinclair", 30, "Architect", "Design-Tech Buyer", "Pro-aesthetic tech."),
    ("Ravi Subramanian", 30, "Quant", "Data-Driven Tech Buyer", "Spreadsheet tracking."),
    ("Brigitte Laurent", 38, "Coach", "Wellness-Tech Buyer", "Recommends clients."),
]

GENERIC_PERSONAS: list[Persona] = [
    ("Mei Lin Zhao", 28, "Marketing Manager", "Wellness-Conscious Pro", "Routine wellness shopper."),
    ("Carlos Mendoza", 40, "Operations Manager", "Family-First Buyer", "Buys for whole household."),
    ("Maya Goldberg", 27, "UX Designer", "Trend-Aware Buyer", "Tries new launches."),
    ("Reginald Cross", 47, "Senior Manager", "Mature Adult Buyer", "Brand-loyal long term."),
    ("Brooke Stephens", 30, "Personal Trainer", "Active Adult", "Performance-focused buyer."),
    ("Brigitte Laurent", 38, "Director", "Premium Buyer", "Top-tier preference."),
    ("Tyler Brennan", 22, "Recent Grad", "Young Adult Buyer", "Budget + quality balance."),
    ("Felix Aurelius", 33, "Engineer", "Optimization Buyer", "Data-driven."),
    ("Naomi Ferguson", 40, "VP", "Working Parent", "Family + career balance."),
    ("Theo Whitaker", 38, "Tech Lead", "Performance Pro", "Cognition + recovery."),
    ("Camille Dupree", 32, "Yoga Studio Manager", "Wellness Lifestyle", "Holistic preferences."),
    ("Hiroshi Tanaka", 35, "Business Owner", "Operator-Athlete", "Test-everything mindset."),
    ("Adaeze Okafor", 27, "PR Director", "Industry-Aware Pro", "Knows the brands."),
    ("Diego Vargas", 28, "Attorney", "Career-Focused Pro", "Convenience priority."),
    ("Talia Rosenberg", 28, "Therapist", "Wellness Practice", "Health-first lifestyle."),
    ("Owen Sinclair", 30, "Architect", "Design-Forward Buyer", "Aesthetic + function."),
    ("Beau Marchetti", 32, "Senior Architect", "Quality-First Buyer", "Investment pieces."),
    ("Marisol Ortega", 35, "Single Parent", "Practical Buyer", "Cost-conscious essentials."),
    ("Trevor Bishop", 35, "IT Manager", "Practical Adult Buyer", "Reliable, no-fuss."),
    ("Linnea Eriksson", 38, "RN", "Healthcare Pro", "Evidence-aware."),
]


# Bank registry for persona_generator.py
BANKS: dict[str, list[Persona]] = {
    # Energy drinks
    "ENERGY_DRINK_STUDENTS": ENERGY_DRINK_STUDENTS,
    "ENERGY_DRINK_FITNESS": ENERGY_DRINK_FITNESS,
    "ENERGY_DRINK_NIGHT_SHIFT": ENERGY_DRINK_NIGHT_SHIFT,
    "ENERGY_DRINK_GAMERS": ENERGY_DRINK_GAMERS,
    # Hydration
    "HYDRATION_ATHLETES": HYDRATION_ATHLETES,
    "HYDRATION_WELLNESS_PROFESSIONALS": HYDRATION_WELLNESS_PROFESSIONALS,
    "HYDRATION_BUSY_PARENTS": HYDRATION_BUSY_PARENTS,
    # Supplements
    "SUPPLEMENTS_GYM_GOERS": SUPPLEMENTS_GYM_GOERS,
    "SUPPLEMENTS_OPTIMIZERS": SUPPLEMENTS_OPTIMIZERS,
    "SUPPLEMENTS_GENERAL_HEALTH": SUPPLEMENTS_GENERAL_HEALTH,
    # Skincare
    "SKINCARE_ENTHUSIASTS": SKINCARE_ENTHUSIASTS,
    "SKINCARE_ACNE_PRONE": SKINCARE_ACNE_PRONE,
    "SKINCARE_ANTI_AGING": SKINCARE_ANTI_AGING,
    # Personal care
    "PERSONAL_CARE_MEN_GROOMING": PERSONAL_CARE_MEN_GROOMING,
    "PERSONAL_CARE_RAZOR_SUBSCRIBERS": PERSONAL_CARE_RAZOR_SUBSCRIBERS,
    "PERSONAL_CARE_BEARD": PERSONAL_CARE_BEARD,
    # NA beer
    "NA_BEER_SOBER_CURIOUS": NA_BEER_SOBER_CURIOUS,
    "NA_BEER_CRAFT_BEER_FANS": NA_BEER_CRAFT_BEER_FANS,
    "NA_BEER_FITNESS": NA_BEER_FITNESS,
    # Coffee alt
    "COFFEE_ALT_WELLNESS": COFFEE_ALT_WELLNESS,
    "COFFEE_ALT_COFFEE_REDUCERS": COFFEE_ALT_COFFEE_REDUCERS,
    "COFFEE_ALT_MUSHROOM_CURIOUS": COFFEE_ALT_MUSHROOM_CURIOUS,
    # Mattress
    "MATTRESS_FIRST_BUYERS": MATTRESS_FIRST_BUYERS,
    "MATTRESS_UPGRADERS": MATTRESS_UPGRADERS,
    "MATTRESS_BACK_PAIN": MATTRESS_BACK_PAIN,
    # Drinkware
    "DRINKWARE_OUTDOOR": DRINKWARE_OUTDOOR,
    "DRINKWARE_OFFICE": DRINKWARE_OFFICE,
    "DRINKWARE_FITNESS": DRINKWARE_FITNESS,
    # Apparel
    "APPAREL_BASICS_MINIMALISTS": APPAREL_BASICS_MINIMALISTS,
    "APPAREL_BASICS_QUALITY_SEEKERS": APPAREL_BASICS_QUALITY_SEEKERS,
    "APPAREL_BASICS_SUSTAINABLE": APPAREL_BASICS_SUSTAINABLE,
    # School supplies
    "SCHOOL_STUDENTS": SCHOOL_STUDENTS,
    "SCHOOL_PARENTS": SCHOOL_PARENTS,
    "SCHOOL_TEACHERS": SCHOOL_TEACHERS,
    "SCHOOL_COLLEGE": SCHOOL_COLLEGE,
    "SCHOOL_AESTHETIC": SCHOOL_AESTHETIC,
    # Religious lifestyle
    "RELIGIOUS_LIFESTYLE_MUSLIM_PROFESSIONALS": RELIGIOUS_LIFESTYLE_MUSLIM_PROFESSIONALS,
    "RELIGIOUS_LIFESTYLE_MUSLIM_FAMILIES": RELIGIOUS_LIFESTYLE_MUSLIM_FAMILIES,
    "RELIGIOUS_LIFESTYLE_SMART_HOME": RELIGIOUS_LIFESTYLE_SMART_HOME,
    # Pet
    "PET_DOG_OWNERS": PET_DOG_OWNERS,
    "PET_CAT_OWNERS": PET_CAT_OWNERS,
    "PET_PREMIUM": PET_PREMIUM,
    # Baby/family
    "BABY_FAMILY_NEW_PARENTS": BABY_FAMILY_NEW_PARENTS,
    "BABY_FAMILY_EXPERIENCED": BABY_FAMILY_EXPERIENCED,
    "BABY_FAMILY_ORGANIC": BABY_FAMILY_ORGANIC,
    # Generic fallbacks
    "FOOD_BEVERAGE_GENERIC": FOOD_BEVERAGE_GENERIC,
    "BEAUTY_PERSONAL_GENERIC": BEAUTY_PERSONAL_GENERIC,
    "HOME_LIFESTYLE_GENERIC": HOME_LIFESTYLE_GENERIC,
    "TECH_WELLNESS_GENERIC": TECH_WELLNESS_GENERIC,
    "GENERIC": GENERIC_PERSONAS,
}


def persona_to_dict(p: Persona) -> dict:
    """Convert tuple persona to dict for JSON output."""
    return {
        "name": p[0],
        "age": p[1],
        "profession": p[2],
        "segment": p[3],
        "profile": p[4],
    }


def get_bank(bank_name: str) -> list[Persona]:
    """Look up a bank by name. Returns GENERIC_PERSONAS if missing."""
    return BANKS.get(bank_name, GENERIC_PERSONAS)


def all_bank_names() -> list[str]:
    """For tests / debug."""
    return list(BANKS.keys())
