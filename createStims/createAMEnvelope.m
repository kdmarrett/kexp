function [final_wheel] = createAMEnvelope(wheel_track, AM_freq, AM_pow, fs)
    if any(AM_freq)
        [m, n] = size(wheel_track);
        t = 0:1/fs:(m/fs + 2); % seconds
        envelope =  (10 ^ (AM_pow/20)).* sin(2.*pi.*AM_freq.*t) + .5;
        envelope = envelope';
        envelope = envelope(1:m, :);
        envelope = createGate(envelope, fs, 1, 1);
        final_wheel(:, 1) = wheel_track(:, 1).* envelope;
        final_wheel(:, 2) = wheel_track(:, 2).* envelope;
    else
        final_wheel = wheel_track;
    end
end

