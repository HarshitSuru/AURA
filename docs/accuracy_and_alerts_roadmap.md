# AURA Accuracy & Alerting Enhancement Roadmap

This roadmap focuses on improving **accuracy** and ensuring alerts are reliable for:
- very high crowd density
- stampede risk
- bottleneck formation
- high movement surges

## High-impact features to add next

1. **Scene calibration profile per camera**
   - Define camera zones (entry, exit, choke points).
   - Store per-camera thresholds for density, motion, and bottleneck.
   - Why: removes one-size-fits-all thresholds and cuts false alerts.

2. **Perspective-aware density estimation**
   - Use perspective map so distant people are weighted correctly.
   - Why: improves people counting in deep scenes.

3. **Track-based crowd flow analytics**
   - Add person tracking (ByteTrack/DeepSORT) and direction vectors.
   - Why: detects counter-flow, stop-and-go, and panic waves better than frame differencing.

4. **Bottleneck map + zone occupancy heatmaps**
   - Draw live heatmap and chokepoint pressure scores.
   - Why: early bottleneck detection before critical congestion.

5. **Multi-level alert policy**
   - Warning → Critical → Emergency with cooldown timers.
   - Why: prevents alert spam and gives operators a clear escalation path.

6. **Adaptive thresholds from baseline behavior**
   - Learn normal density and movement by hour/day.
   - Why: raises alerts only when conditions exceed expected baseline.

7. **Sensor fusion options**
   - Optional turnstile count, LiDAR, Wi-Fi probe, or thermal feed fusion.
   - Why: improves confidence under occlusion and poor lighting.

8. **Alert confidence score**
   - Combine density, movement, bottleneck, and trend slope into one risk score.
   - Why: users can trust alerts with transparent confidence levels.

9. **Post-event replay and root-cause timeline**
   - Save 30–60s pre-alert clips and annotated metrics timeline.
   - Why: helps validate true positives and tune thresholds quickly.

10. **Operator action workflows**
    - One-click SOP prompts (open gate, redirect flow, call security).
    - Why: faster response after alert generation.

## Sound alert recommendations

- Use **distinct system sounds per event type**:
  - High Density: short repeating beep.
  - High Movement: double beep.
  - Bottleneck: pulsed tone.
  - Stampede Risk: continuous urgent siren.
- Add **minimum repeat interval** (e.g., 15s) per event type.
- Add **silent mode + visual-only mode** for night operations.
- Add **audio device health check** at startup.

## Validation plan for high accuracy

- Create labeled clips for each scenario: normal, high density, bottleneck, stampede.
- Track precision/recall per alert type.
- Tune thresholds camera-by-camera weekly.
- Run stress tests at different FPS and lighting conditions.
- Keep a false-positive/false-negative review log for model improvements.
