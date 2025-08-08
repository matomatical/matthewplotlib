import numpy as np
import einops
import matthewplotlib as mp

image_data = """
    deeeeeefeffffeccddeeeedeeeeeeeee eeeeeeffffec9755777a9abdeeeeeeee
    deeeefefec85555566656778aceeeeee deeeeeeea55555545565667768aeeeee
    deeeefec5554443444455467789ceeee ceeeeca7544443333233345457bceeee
    ceeeed754444333222323333459ceeee cdeccc544433333322222334567ceeee
    bdcaa7443356788666422343465cfeee bcb776433479999999852235354bfeee
    bca8853324999aaaaa9853345439feee ba96423326aaaaaaaaaa75325437ddee
    c85324432899aabbabbba8433435abdd ca53453148655799aaabaa6134547bbd
    ca6244305865346886446992346669be ca7244306976456aa54578943557bdee
    cb7244205998878bc979bbb51467bddd cc82342059a9999adcabcdc53465698d
    cc93332039aa989bdcbccdc52443207d cca42321299a889accabcdc3234356be
    cba52221089888589bbacdb0233328de cb963221089997468abbbda022355cfe
    ddcb532106985456887acd70228a9dee ddda7521049865679aaacc20217deeee
    ded9a701007888779cccd902122aeeee ddd9b802104888768accc301114adfee
    dddbb83021267888abcc6021135aeeee deedca2540365689aaa90230335ceeee
    eeeecc963227754667ba026505dfeeee cbaba964432688779bcb34646deeeefe
    6665435553357899bbc925877999acdd 66655454455578aaaba6349988666779
""".strip().split()
image = np.asarray([[int(a,base=16) for a in row] for row in image_data])

plot = mp.wrap(
    mp.image(image / 15),
    mp.image(image / 15, colormap=mp.viridis),
    mp.image(1 - image / 15, colormap=mp.cool),
    mp.image(image / 15, colormap=mp.magentas),
    mp.image(image // 4 + 8, colormap=mp.sweetie16),
    mp.image(image // 4, colormap=mp.sweetie16),
)

print("printing plot...")
print(plot)
print("saving to 'out.png'...")
plot.saveimg('out.png')
