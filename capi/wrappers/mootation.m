% SPDX-License-Identifier: Apache-2.0
%
% MOOtation from MATLAB (and Octave), through the C ABI.
%
%     zdt1 = @(x) [x(1), (1 + 9*sum(x(2:end))/(numel(x)-1)) * ...
%                        (1 - sqrt(x(1) / (1 + 9*sum(x(2:end))/(numel(x)-1))))];
%
%     settings = sprintf(['algorithm = nsga2\n' ...
%                         'pop_size  = 40\n' ...
%                         'max_gen   = 100\n' ...
%                         'n_vars    = 10\n' ...
%                         'n_objs    = 2\n' ...
%                         'lower     = 0\n' ...
%                         'upper     = 1\n']);
%
%     [X, F, cv] = mootation('minimize', zdt1, settings, ...
%                            'library', 'libmootation', ...
%                            'header',  'capi/mootation.h');
%     plot(F(:,1), F(:,2), 'o')
%
% loadlibrary needs the header, and it will not accept capi.h directly because
% of the dllimport/visibility macros. Generate a plain one first:
%
%     cpp -P -DMOOTATION_C_BUILD= -D'MOO_API=' include/mootation/capi.h > capi/mootation.h
%
% or copy the prototypes by hand — there are sixteen of them.
%
% MATLAB is column-major and the ABI is row-major, so every reshape here
% transposes. It is deliberate and confined to this file.

function varargout = mootation(action, varargin)
    switch lower(action)
        case 'minimize'
            [varargout{1:nargout}] = do_minimize(varargin{:});
        case 'algorithms'
            varargout{1} = do_algorithms(varargin{:});
        case 'version'
            varargout{1} = do_version(varargin{:});
        otherwise
            error('mootation:action', ...
                  'unknown action "%s" (minimize | algorithms | version)', action);
    end
end

% ── Library handling ────────────────────────────────────────────────────────

function ensure_loaded(libname, header)
    if ~libisloaded(libname)
        loadlibrary(libname, header);
    end
end

function [libname, header, rest] = take_lib_opts(args)
    libname = 'libmootation';
    header  = 'mootation.h';
    rest    = {};
    k = 1;
    while k <= numel(args)
        if ischar(args{k}) && strcmpi(args{k}, 'library')
            libname = args{k+1}; k = k + 2;
        elseif ischar(args{k}) && strcmpi(args{k}, 'header')
            header = args{k+1}; k = k + 2;
        else
            rest{end+1} = args{k}; %#ok<AGROW>
            k = k + 1;
        end
    end
end

% ── Public actions ──────────────────────────────────────────────────────────

function v = do_version(varargin)
    [lib, hdr] = take_lib_opts(varargin);
    ensure_loaded(lib, hdr);
    v = calllib(lib, 'moo_version_string');
end

function names = do_algorithms(varargin)
    [lib, hdr] = take_lib_opts(varargin);
    ensure_loaded(lib, hdr);
    n = calllib(lib, 'moo_algorithm_count');
    if n < 0, error('mootation:abi', 'moo_algorithm_count failed'); end
    names = cell(n, 1);
    for i = 0:(n-1)
        names{i+1} = calllib(lib, 'moo_algorithm_name', int32(i));
    end
end

function [X, F, cv] = do_minimize(fn, settings, varargin)
    [lib, hdr, rest] = take_lib_opts(varargin);
    ensure_loaded(lib, hdr);

    cons = [];
    for k = 1:2:numel(rest)
        if strcmpi(rest{k}, 'constraints'), cons = rest{k+1}; end
    end

    s = calllib(lib, 'moo_open', settings);
    if isNull(s)
        error('mootation:open', '%s', last_error(lib, s));
    end
    cleanup = onCleanup(@() calllib(lib, 'moo_close', s));

    nv = calllib(lib, 'moo_n_vars', s);
    no = calllib(lib, 'moo_n_objs', s);
    nc = calllib(lib, 'moo_n_cons', s);

    n = check(lib, s, calllib(lib, 'moo_ask_count', s));
    while n > 0
        % The batch size is the ALGORITHM's choice, never pop_size.
        xbuf = zeros(n * nv, 1);
        [~, xbuf] = calllib(lib, 'moo_ask', s, xbuf, int32(numel(xbuf)));
        X = reshape(xbuf, nv, n).';        % row-major in, transpose out

        Fb = zeros(n, no);
        for i = 1:n
            Fb(i, :) = reshape(fn(X(i, :)), 1, no);
        end
        fbuf = reshape(Fb.', [], 1);

        if nc > 0
            if isempty(cons)
                error('mootation:constraints', ...
                      'this run has %d constraints; pass ''constraints'', @fn', nc);
            end
            Gb = zeros(n, nc);
            for i = 1:n
                Gb(i, :) = reshape(cons(X(i, :)), 1, nc);
            end
            gbuf = reshape(Gb.', [], 1);
            n = check(lib, s, calllib(lib, 'moo_tell', s, ...
                      fbuf, int32(numel(fbuf)), gbuf, int32(numel(gbuf))));
        else
            n = check(lib, s, calllib(lib, 'moo_tell', s, ...
                      fbuf, int32(numel(fbuf)), libpointer('doublePtr'), int32(0)));
        end
    end

    m  = check(lib, s, calllib(lib, 'moo_result_count', s));
    xb = zeros(m * nv, 1);
    fb = zeros(m * no, 1);
    cb = zeros(m, 1);
    [~, xb, fb, cb] = calllib(lib, 'moo_result', s, ...
        xb, int32(numel(xb)), fb, int32(numel(fb)), cb, int32(m));

    X  = reshape(xb, nv, m).';
    F  = reshape(fb, no, m).';
    cv = cb;
end

% ── Helpers ─────────────────────────────────────────────────────────────────

function v = check(lib, s, v)
    if v < 0
        error('mootation:abi', '%s', last_error(lib, s));
    end
end

function msg = last_error(lib, s)
    msg = calllib(lib, 'moo_last_error', s);
    if isempty(msg), msg = 'unknown error'; end
end

function tf = isNull(p)
    tf = isempty(p) || (isa(p, 'lib.pointer') && p.isNull);
end
