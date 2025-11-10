# Machine Learning Lønsforudsigelse - Dokumentation

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/6b05152b-51a8-4715-a8e8-adf7f68f1a8a" />

## Input (Features) og Output (Target)

I machine learning skelnes der mellem input og output data. **Input** eller **features** er de variabler, der bruges som grundlag for forudsigelser. **Output** eller **target** er den værdi, der ønskes forudsagt.

I dette program er:
- **Feature**: `YearsExperience` - antal års erhvervserfaring
- **Target**: `Salary` - løn i kroner

Programmet bruger års erfaring som eneste feature til at forudsige lønniveau.

## Regressionslinje, Hældning og Intercept

Regressionslinjen repræsenterer den matematiske sammenhæng mellem feature og target. Linjen defineres af ligningen: `ŷ = 9449.962 * YearsExperience + 24848.204`

**Hældning (9449.962)**: Angiver lønstigning per års erfaring. Hver ekstra års erfaring resulterer i en gennemsnitlig lønforhøjelse på 9.450 kr.

**Intercept (24848.204)**: Repræsenterer forventet løn ved 0 års erfaring - grundlønnen på cirka 24.848 kr.

## Model Træning

Model træning refererer til processen, hvor algoritmen lærer mønstre fra eksisterende data. I koden udføres dette med:

```python
model = LinearRegression().fit(X, y)
```

Under træningen:
1. Algoritmen analyserer alle datapunkter (erfaring og tilhørende løn)
2. Beregner den optimale regressionslinje ved at minimere afstanden mellem linje og datapunkter
3. Fastlægger hældning og intercept for bedste mulige tilpasning

Efter træning kan modellen generere forudsigelser for nye data.

## Praktisk Anvendelse

Modellen har flere potentielle anvendelsesområder:

**HR og rekruttering**:
- Benchmarking af lønstrukturer mod markedet
- Fastsættelse af konkurrencedygtige startlønninger
- Budgetplanlægning baseret på medarbejderprofiler

**Karriereplanlægning**:
- Estimering af lønudvikling over tid
- Vurdering af værdi af erhvervserfaring
- Forhandlingsgrundlag ved jobskifte

**Eksempel på anvendelse**:
```python
years = 4.5
predicted_salary = model.predict([[years]])
# Resultat: 67.373 kr for person med 4,5 års erfaring
```

**Begrænsninger**:
- Modellen antager lineær sammenhæng mellem erfaring og løn
- Tager ikke højde for faktorer som uddannelse, branche eller geografi
- Kvalitet afhænger af træningsdataets repræsentativitet
- Kræver regelmæssig opdatering med nye data

Modellen gemmes med `joblib.dump()` for genbrug uden genoptræning.
