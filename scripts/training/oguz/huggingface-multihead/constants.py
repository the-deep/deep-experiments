SECTORS = [
    "Agriculture",
    "Cross",
    "Education",
    "Food Security",
    "Health",
    "Livelihoods",
    "Logistics",
    "Nutrition",
    "Protection",
    "Shelter",
    "WASH",
]

PILLARS_1D = [
    "Context",
    "Shock/Event",
    "Casualties",
    "Displacement",
    "Humanitarian Access",
    "Information And Communication",
    "Covid-19",
]

SUBPILLARS_1D = [
    [
        "Context->Demography",
        "Context->Economy",
        "Context->Environment",
        "Context->Security & Stability",
        "Context->Socio Cultural",
        "Context->Legal & Policy",
        "Context->Politics",
        "Context->Technological",
    ],
    [
        "Shock/Event->Type And Characteristics",
        "Shock/Event->Underlying/Aggravating Factors",
        "Shock/Event->Hazard & Threats",
    ],
    ["Casualties->Dead", "Casualties->Injured", "Casualties->Missing"],
    [
        "Displacement->Type/Numbers/Movements",
        "Displacement->Push Factors",
        "Displacement->Pull Factors",
        "Displacement->Intentions",
        "Displacement->Local Integration",
    ],
    [
        "Humanitarian Access->Relief To Population",
        "Humanitarian Access->Population To Relief",
        "Humanitarian Access->Physical Constraints",
        (
            "Humanitarian Access->Number Of People Facing Humanitarian Access Constraints"
            "/Humanitarian Access Gaps"
        ),
    ],
    [
        "Information And Communication->Information Challenges And Barriers",
        "Information And Communication->Communication Means And Preferences",
        "Information And Communication->Knowledge And Info Gaps (Pop)",
        "Information And Communication->Knowledge And Info Gaps (Hum)",
    ],
    [
        "Covid-19->Cases",
        "Covid-19->Deaths",
        "Covid-19->Testing",
        "Covid-19->Contact Tracing",
        "Covid-19->Hospitalization & Care",
        "Covid-19->Vaccination",
        "Covid-19->Restriction Measures",
    ],
]

PILLARS_2D = [
    "Humanitarian Conditions",
    "Capacities & Response",
    "Impact",
    "Priority Interventions",
    "At Risk",
    "Priority Needs",
]

SUBPILLARS_2D = [
    [
        "Humanitarian Conditions->Coping Mechanisms",
        "Humanitarian Conditions->Living Standards",
        "Humanitarian Conditions->Physical And Mental Well Being",
        "Humanitarian Conditions->Number Of People In Need",
    ],
    [
        "Capacities & Response->International Response",
        "Capacities & Response->National Response",
        "Capacities & Response->Local Response",
        "Capacities & Response->Number Of People Reached/Response Gaps",
    ],
    [
        "Impact->Driver/Aggravating Factors",
        "Impact->Impact On People",
        "Impact->Impact On Systems, Services And Networks",
        "Impact->Number Of People Affected",
    ],
    [
        "Priority Interventions->Expressed By Humanitarian Staff",
        "Priority Interventions->Expressed By Population",
    ],
    [
        "At Risk->Risk And Vulnerabilities",
        "At Risk->Number Of People At Risk",
    ],
    [
        "Priority Needs->Expressed By Humanitarian Staff",
        "Priority Needs->Expressed By Population",
    ],
]
